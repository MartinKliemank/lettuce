import torch.distributed as dist
from timeit import default_timer as timer
from lettuce import (
    LettuceException, StandardStreaming, Simulation
)
from lettuce.util import pressure_poisson
import pickle
from copy import deepcopy
import warnings
import torch
import numpy as np

__all__ = ["DistributedSimulation", "DistributedStreaming"]

class DistributedSimulation(Simulation):

    def __init__(self, flow, lattice, collision, streaming, rank, size):
        self.rank = rank
        self.size = size

        self.flow = flow
        self.lattice = lattice
        self.collision = collision
        self.streaming = streaming
        self.i = 0

        grid = [x[int(np.floor(flow.grid[0].shape[0]*rank/size)):int(np.floor(flow.grid[0].shape[0]*(rank+1)/size)),...] for x in flow.grid]
        print(f"Process {rank} covers {int(np.floor(flow.grid[0].shape[0]*rank/size))}:{int(np.floor(flow.grid[0].shape[0]*(rank+1)/size))}!")
        p, u = flow.initial_solution(grid)
        assert list(p.shape) == [1] + list(grid[0].shape), \
            LettuceException(f"Wrong dimension of initial pressure field. "
                             f"Expected {[1] + list(grid[0].shape)}, "
                             f"but got {list(p.shape)}.")
        assert list(u.shape) == [lattice.D] + list(grid[0].shape), \
            LettuceException("Wrong dimension of initial velocity field."
                             f"Expected {[lattice.D] + list(grid[0].shape)}, "
                             f"but got {list(u.shape)}.")
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        self.f = lattice.equilibrium(rho, lattice.convert_to_tensor(u))

        self.reporters = []

        # Define masks, where the collision or streaming are not applied
        x = grid
        self.no_collision_mask = lattice.convert_to_tensor(np.zeros_like(x[0], dtype=bool))
        no_stream_mask = lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool))

#find out which boundaries apply in what region... (input large f and take small part of mask?) (NOT TESTED YET!!!!!!)

        # Apply boundaries
        self._boundaries = deepcopy(self.flow.boundaries)  # store locally to keep the flow free from the boundary state
        for boundary in self._boundaries:
            if hasattr(boundary, "make_no_collision_mask"):
                self.no_collision_mask = self.no_collision_mask | boundary.make_no_collision_mask(torch.zeros([lattice.Q]+list(flow.grid[0].shape)))[int(np.floor(flow.grid[0].shape[0]*rank/size)):int(np.floor(flow.grid[0].shape[0]*(rank+1)/size)),...]
            if hasattr(boundary, "make_no_stream_mask"):
                no_stream_mask = no_stream_mask | boundary.make_no_stream_mask(torch.zeros([lattice.Q]+list(flow.grid[0].shape)))[:,int(np.floor(flow.grid[0].shape[0]*rank/size)):int(np.floor(flow.grid[0].shape[0]*(rank+1)/size)),...]
        if no_stream_mask.any():
            self.streaming.no_stream_mask = no_stream_mask

    def step(self, num_steps):
        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        start = timer()
        if self.i == 0:
            self._report()
        for _ in range(num_steps):
            self.i += 1
            self.f = self.streaming(self.f)
            # Perform the collision routine everywhere, expect where the no_collision_mask is true
            self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))
            for boundary in self._boundaries:
                self.f = boundary(self.f)
            self._report()
        end = timer()
        seconds = end - start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups


class DistributedStreaming(StandardStreaming):
    """Standard streaming for distributed simulation, domain is separated along 0th (x)-dimension"""

    def __init__(self, lattice, rank, size):
        self.lattice = lattice
        self.size = size
        self.rank = rank
        self.prev = self.rank - 1 if self.rank != 0 else self.size - 1
        self.next = self.rank + 1 if self.rank != self.size - 1 else 0
        self._no_stream_mask = None

#maybe make stream only roll again and stream all I's for one direction at the same time?
    def _stream(self, f, i):
        if self.lattice.e[i, 0] != 0:
            if 1:
                f = f[i]
                output = (f[-1, ...] if self.lattice.e[i, 0] > 0 else f[0, ...]).contiguous()
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
            if 0:
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
