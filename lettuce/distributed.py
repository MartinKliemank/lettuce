import torch.distributed as dist
from timeit import default_timer as timer
from lettuce import (
    LettuceException, StandardStreaming, Simulation, AntiBounceBackOutlet
)
import pickle
from copy import deepcopy
import torch
import numpy as np
import os


__all__ = ["DistributedSimulation", "DistributedStreaming", "DistributedStreamcolliding"]


class DistributedSimulation(Simulation):

    def __init__(self, flow, lattice, collision, streaming, rank, size):
        self.rank = rank
        self.size = size

        self.flow = flow
        self.lattice = lattice
        self.collision = collision
        self.streaming = streaming
        self.i = 0

        self.index = [flow.grid.index, ...]

        print(flow.grid.shape)
        print(f"Process {self.rank} covers {self.index}")
        p, u = flow.initial_solution(flow.grid())
        assert list(p.shape) == [1] + list(flow.grid.shape), \
            LettuceException(f"Wrong dimension of initial pressure field. "
                             f"Expected {[1] + list(flow.grid.shape)}, "
                             f"but got {list(p.shape)}.")
        assert list(u.shape) == [lattice.D] + list(flow.grid.shape), \
            LettuceException("Wrong dimension of initial velocity field."
                             f"Expected {[lattice.D] + list(flow.grid.shape)}, "
                             f"but got {list(u.shape)}.")
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        self.f = lattice.equilibrium(rho, u)

        self.reporters = []

        # Define masks, where the collision or streaming are not applied
        self.no_collision_mask = lattice.convert_to_tensor(np.zeros_like(flow.grid()[0], dtype=bool))
        no_stream_mask = lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool))

        # Apply boundaries
        self._boundaries = deepcopy(self.flow.boundaries)  # store locally to keep the flow free from the boundary state
        for boundary in self._boundaries:
            if hasattr(boundary, "make_no_collision_mask"):
                self.no_collision_mask = self.no_collision_mask | flow.grid.select(boundary.make_no_collision_mask(flow.grid.global_shape))#[self.index] #WIP index weg / anpassen
            if hasattr(boundary, "make_no_stream_mask"):
                no_stream_mask = no_stream_mask | flow.grid.select(boundary.make_no_stream_mask(torch.Size([lattice.Q]+list(flow.grid.global_shape))))#[[slice(None)] + self.index]
        if no_stream_mask.any():
            self.streaming.no_stream_mask = no_stream_mask

    def step(self, num_steps):
        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        start = timer()
        if self.i == 0:
            self._report()
        for _ in range(num_steps):
            self.i += 1
            if isinstance(self.streaming, DistributedStreamcolliding):
                self.f = self.streaming(self.f, self.no_collision_mask)
            else:
                self.f = self.streaming(self.f)
                # Perform the collision routine everywhere, expect where the no_collision_mask is true
                self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))
                for boundary in self._boundaries:
                    # Unterscheidung in "has direction" und has mask -> indices vs. ifs
                    if isinstance(boundary, AntiBounceBackOutlet):
                        if boundary.direction[0] == -1 and self.rank == 0:
                            self.f = boundary(self.f)
                        elif boundary.direction[0] == 1 and self.rank == self.size - 1:
                            self.f = boundary(self.f)
                        elif boundary.direction[0] == 0:
                            self.f = boundary(self.f)
                    else:
                        self.f = boundary(self.f)
            self._report()
        end = timer()
        seconds = end - start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups

    def save_checkpoint(self, filename):
        """Write f as np.array using pickle module."""
        directory, file = os.path.split(filename)
        file = f"{self.rank}-" + file
        filename = os.path.join(directory, file)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        with open(filename, "wb") as fp:
            pickle.dump(self.f, fp)

    def load_checkpoint(self, filename):
        """Load f as np.array using pickle module."""
        folder, file = os.path.split(filename)
        file = f"{self.rank}-" + file
        filename = os.path.join(folder, file)
        with open(filename, "rb") as fp:
            self.f = pickle.load(fp)

class DistributedStreaming(StandardStreaming):
    """Standard streaming for distributed simulation, domain is separated along 0th (x)-dimension"""

    def __init__(self, lattice, rank, size):
        self.lattice = lattice
        self.size = size
        self.rank = rank
        self.prev = self.rank - 1 if self.rank != 0 else self.size - 1
        self.next = self.rank + 1 if self.rank != self.size - 1 else 0
        self.forward = np.argwhere(self.lattice.stencil.e[:, 0] > 0)
        self.rest = np.argwhere(self.lattice.stencil.e[:, 0] == 0)
        self.backward = np.argwhere(self.lattice.stencil.e[:, 0] < 0)
        self.no_stream_mask = None

    def __call__(self, f):
        if 0:
            forward = np.argwhere(self.lattice.e[:, 0] > 0)
            backward = np.argwhere(self.lattice.e[:, 0] < 0)
            output_forward = f[forward, -1, ...].detach().clone().contiguous()
            output_backward = f[backward, 0, ...].detach().clone().contiguous()
            input_forward = torch.zeros_like(f[forward, 0, ...])
            input_backward = torch.zeros_like(f[backward, 0, ...])
            if self.rank % 2 == 0:
                dist.send(tensor=output_forward, dst=self.next)
                dist.recv(tensor=input_forward.contiguous(), src=self.prev)
                dist.send(tensor=output_backward, dst=self.prev)
                dist.recv(tensor=input_backward.contiguous(), src=self.next)
            else:
                dist.recv(tensor=input_forward.contiguous(), src=self.prev)
                dist.send(tensor=output_forward, dst=self.next)
                dist.recv(tensor=input_backward.contiguous(), src=self.next)
                dist.send(tensor=output_backward, dst=self.prev)

            f = torch.cat((torch.zeros_like(f[:, 0, ...]).unsqueeze(1), f, torch.zeros_like(f[:, 0, ...]).unsqueeze(1)), dim=1)
            f[forward, 0, ...] = input_forward
            f[backward, -1, ...] = input_backward

            for i in range(1, self.lattice.Q):
                if self.no_stream_mask is None:
                    f[i] = self._stream(f, i)
                else:
                    new_fi = self._stream(f, i)
                    f[i] = torch.where(self.no_stream_mask[i], f[i], new_fi)

            return f[:, 1:-1, ...]
        else:
            output_forward = f[self.forward, -1, ...].detach().clone().contiguous()
            output_backward = f[self.backward, 0, ...].detach().clone().contiguous()
            input_forward = torch.zeros_like(f[self.forward, 0, ...])
            input_backward = torch.zeros_like(f[self.backward, 0, ...])

            outf = dist.isend(tensor=output_forward, dst=self.next)
            outb = dist.isend(tensor=output_backward, dst=self.prev)
            inf = dist.irecv(tensor=input_forward.contiguous(), src=self.prev)
            inb = dist.irecv(tensor=input_backward.contiguous(), src=self.next)

            f = torch.cat((torch.zeros_like(f[:, 0, ...]).unsqueeze(1), f, torch.zeros_like(f[:, 0, ...]).unsqueeze(1)), dim=1)
            if self.no_stream_mask is not None:
                no_stream_mask = torch.cat((torch.zeros_like(self.no_stream_mask[:, 0, ...]).unsqueeze(1),
                                            self.no_stream_mask,
                                            torch.zeros_like(self.no_stream_mask[:, -1, ...]).unsqueeze(1)), dim=1)
            inf.wait()
            #WIP: vor diesem wait schon mal rest streamen?
            f[self.forward, 0, ...] = input_forward
            inb.wait()
            f[self.backward, -1, ...] = input_backward
            for i in range(1, self.lattice.Q):
                i = int(i)
                if self.no_stream_mask is None:
                    f[i] = self._stream(f, i)
                else:
                    new_fi = self._stream(f, i)
                    f[i] = torch.where(no_stream_mask[i], f[i], new_fi)
            ## Alternative that does backward / forward in the order the data arrive (doesn't work because .is_completed() is apparently bugged? ( https://github.com/pytorch/pytorch/issues/30723 )
            # flag = 0
            # while(flag < 3):
            #     if flag == 0:
            #         for i in self.rest[0]:
            #             i = int(i)
            #             if self.no_stream_mask is None:
            #                 f[i] = self._stream(f, i)
            #             else:
            #                 new_fi = self._stream(f, i)
            #                 f[i] = torch.where(self.no_stream_mask[i], f[i], new_fi)
            #         flag = flag + 1
            #     if inf.is_completed():
            #         f[forward, 0, ...] = input_forward
            #         for i in self.forward[0]:
            #             i = int(i)
            #             if self.no_stream_mask is None:
            #                 f[i] = self._stream(f, i)
            #             else:
            #                 new_fi = self._stream(f, i)
            #                 f[i] = torch.where(self.no_stream_mask[i], f[i], new_fi)
            #         flag = flag + 1
            #     if inb.is_completed():
            #         f[backward, -1, ...] = input_backward
            #         for i in self.backward[0]:
            #             i = int(i)
            #             if self.no_stream_mask is None:
            #                 f[i] = self._stream(f, i)
            #             else:
            #                 new_fi = self._stream(f, i)
            #                 f[i] = torch.where(self.no_stream_mask[i], f[i], new_fi)
            #         flag = flag + 1
            outf.wait()
            outb.wait()
            return f[:, 1:-1, ...]

    def _Diststream(self, f, i):
        if self.lattice.e[i, 0] != 0 and self.size > 1:
            if 0:
                f = f[i]
                output = (f[-1, ...] if self.lattice.e[i, 0] > 0 else f[0, ...]).detach().clone().contiguous()
                if self.rank != 0:
                    dist.send(tensor=output,
                              dst=self.next if self.lattice.e[i, 0] > 0 else self.prev)
                f = torch.cat((torch.zeros_like(f[0, ...]).unsqueeze(0), f, torch.zeros_like(f[0, ...]).unsqueeze(0)), dim=0)
                input = torch.zeros_like(f[0, ...])
                dist.recv(tensor=input.contiguous(),
                          src=self.prev if self.lattice.e[i, 0] > 0 else self.next)
                if self.lattice.e[i, 0] > 0:
                    f[0, ...] = input
                else:
                    f[-1, ...] = input
                f = torch.roll(f, shifts=tuple(self.lattice.stencil.e[i]), dims=tuple(np.arange(self.lattice.D)))
                if self.rank == 0:
                    dist.send(tensor=output,
                              dst=self.next if self.lattice.e[i, 0] > 0 else self.prev)
                return f[1:-1, ...]
            if 1:
                f = f[i]
                output = (f[-1, ...] if self.lattice.e[i, 0] > 0 else f[0, ...]).detach().clone().contiguous()
                out = dist.isend(tensor=output,
                          dst=self.next if self.lattice.e[i, 0] > 0 else self.prev)
                f = torch.cat((torch.zeros_like(f[0, ...]).unsqueeze(0), f, torch.zeros_like(f[0, ...]).unsqueeze(0)), dim=0)
                input = torch.zeros_like(f[0, ...])
                inp = dist.irecv(tensor=input.contiguous(),
                          src=self.prev if self.lattice.e[i, 0] > 0 else self.next)
                inp.wait()
                if self.lattice.e[i, 0] > 0:
                    f[0, ...] = input
                else:
                    f[-1, ...] = input
                f = torch.roll(f, shifts=tuple(self.lattice.stencil.e[i]), dims=tuple(np.arange(self.lattice.D)))
                out.wait()
                return f[1:-1, ...]
        else:
            return torch.roll(f[i], shifts=tuple(self.lattice.stencil.e[i]), dims=tuple(np.arange(self.lattice.D)))


class DistributedStreamcolliding(object):

    def __init__(self, lattice, rank, size, collision):
        self.lattice = lattice
        self.size = size
        self.rank = rank
        self.prev = self.rank - 1 if self.rank != 0 else self.size - 1
        self.next = self.rank + 1 if self.rank != self.size - 1 else 0
        self.forward = np.argwhere(self.lattice.stencil.e[:, 0] > 0)
        self.rest = np.argwhere(self.lattice.stencil.e[:, 0] == 0)
        self.backward = np.argwhere(self.lattice.stencil.e[:, 0] < 0)
        self._no_stream_mask = None
        self.collision = collision

    @property
    def no_stream_mask(self):
        return self._no_stream_mask

    @no_stream_mask.setter
    def no_stream_mask(self, mask):
        self._no_stream_mask = mask

    def __call__(self, f, no_collision_mask):
        # in x direction: (> positive)
        #   Input_Forward > |Domain| > Output_Forward
        # Output_Backward < |Domain| < Input_Backward


        output_forward = f[self.forward, -1, ...].detach().clone().contiguous()
        output_backward = f[self.backward, 0, ...].detach().clone().contiguous()
        input_forward = torch.zeros_like(f[self.forward, 0, ...])
        input_backward = torch.zeros_like(f[self.backward, -1, ...])

        outf = dist.isend(tensor=output_forward, dst=self.next)
        outb = dist.isend(tensor=output_backward, dst=self.prev)
        inf = dist.irecv(tensor=input_forward.contiguous(), src=self.prev)
        inb = dist.irecv(tensor=input_backward.contiguous(), src=self.next)


        # hier inneres Streamen + colliden, dann unten nur noch Rand streamen und colliden

        """
        for boundary in self._boundaries:
            # Unterscheidung in "has direction" und has mask -> indices vs. ifs
            if isinstance(boundary, AntiBounceBackOutlet):
                if boundary.direction[0] == -1 and self.rank == 0:
                    self.f = boundary(self.f)
                elif boundary.direction[0] == 1 and self.rank == self.size - 1:
                    self.f = boundary(self.f)
                elif boundary.direction[0] == 0:
                    self.f = boundary(self.f)
            else:
                self.f = boundary(self.f, self.index)
        """

        # normal streaming, because fs that move out on one side will end up in the place where the incoming ones will overwrite them?
        for i in range(1, self.lattice.Q):
            i = int(i)
            if self.no_stream_mask is None:
                f[i] = self._stream(f[i], i)
            else:
                new_fi = self._stream(f[i], i)
                f[i] = torch.where(self.no_stream_mask[i], f[i], new_fi)
        # collision on full domain - first and last x-coordinate where the wrong fs are
        f[:, 1:-1, ...] = torch.where(no_collision_mask[1:-1, ...], f[:, 1:-1, ...], self.collision(f[:, 1:-1, ...]))
        inf.wait()
        # boundaries need to be watched out for!!!
        input_forward = torch.cat((input_forward, torch.zeros_like(input_forward)), dim=1)
        for i in self.forward[:, 0]:
            i = int(i)
            input_forward[np.argwhere(self.forward == i)[0][0]] = self._stream(input_forward[np.argwhere(self.forward == i)[0][0]], i)
        f[self.forward, 0, ...] = input_forward[:, -1, ...].unsqueeze(1)
        #print(f"Process {self.rank} uses input as {input_forward[:, -1, ...]}")
        f[:, 0, ...] = torch.where(no_collision_mask[0, ...].unsqueeze(0), f[:, 0, ...].unsqueeze(1), self.collision(f[:, 0, ...].unsqueeze(1))).squeeze(1)
        inb.wait()
        # boundaries need to be watched out for!!!
        input_backward = torch.cat((torch.zeros_like(input_backward), input_backward), dim=1)
        for i in self.backward[:, 0]:
            i = int(i)
            input_backward[np.argwhere(self.backward == i)[0][0]] = self._stream(input_backward[np.argwhere(self.backward == i)[0][0]], i)
        f[self.backward, -1, ...] = input_backward[:, 0, ...].unsqueeze(1)

        f[:, -1, ...] = torch.where(no_collision_mask[-1, ...].unsqueeze(0), f[:, -1, ...].unsqueeze(1), self.collision(f[:, -1, ...].unsqueeze(1))).squeeze(1)

        outf.wait()
        outb.wait()
        return f

    def _stream(self, f, i):
        return torch.roll(f, shifts=tuple(self.lattice.stencil.e[i]), dims=tuple(np.arange(self.lattice.D)))