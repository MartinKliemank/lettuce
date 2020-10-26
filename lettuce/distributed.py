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

class DistributedSimulation(Simulation):

    def __init__(self, flow, lattice, collision, streaming, rank, size):
        self.rank = rank
        self.size = size
        self.prev = self.rank - 1 if self.rank != 0 else self.size - 1
        self.next = self.rank + 1 if self.rank != self.size - 1 else 0

        self.flow = flow
        self.lattice = lattice
        self.collision = collision
        self.streaming = streaming
        self.i = 0

        grid = flow.grid[int(flow.grid.size(0)*rank/size):int(flow.grid.size(0)*(rank-1)/size)]
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
        self.no_collision_mask = lattice.convert_to_tensor(np.zeros_like(x[0],dtype=bool))
        no_stream_mask = lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool))

#find out which boundaries apply in what region... (input large f and take small part of mask?)

        # Apply boundaries
        self._boundaries = deepcopy(self.flow.boundaries)  # store locally to keep the flow free from the boundary state
        for boundary in self._boundaries:
            if hasattr(boundary, "make_no_collision_mask"):
                self.no_collision_mask = self.no_collision_mask | boundary.make_no_collision_mask(self.f.shape)
            if hasattr(boundary, "make_no_stream_mask"):
                no_stream_mask = no_stream_mask | boundary.make_no_stream_mask(self.f.shape)
        if no_stream_mask.any():
            self.streaming.no_stream_mask = no_stream_mask

    def step(self, num_steps):

        text = 3


class DistributedStreaming(StandardStreaming):
    """Standard streaming for distributed simulation, domain is separated along 0th (x)-dimension"""

    def __init__(self, lattice, prev, rank, next):
        self.lattice = lattice
        self.prev = prev
        self.rank = rank
        self.next = next
        self._no_stream_mask = None

    def _stream(self, f, i):
        if self.rank == 0:
            dist.send(tensor=(f[-1, ...] if self.lattice.e[i, 0] > 0 else f[0, ...]).contiguous(),
                      dst=self.next if self.lattice.e[i, 0] > 0 else self.prev)

        # add new columns (add 1 coordinate in x at front and back of domain for in / outflow)
        f = torch.cat((torch.zeros_like(f[i, 0, ...]).unsqueeze(0), f[i], torch.zeros_like(f[i, 0, ...]).unsqueeze(0)), dim=0)
        # receive inflow from prev / next process into first / last column
        dist.recv(tensor=(f[0, ...] if self.lattice.e[i, 0] > 0 else f[-1, ...]).contiguous(),
                  src=self.prev if self.lattice.e[i, 0] > 0 else self.next)
        # stream normally in f
        f = torch.roll(f, shifts=tuple(self.lattice.stencil.e[i]), dims=tuple(np.arange(self.lattice.D)))
        # stream out the outflow that has ended up in the overhanging column
        if self.rank != 0:
            dist.send(tensor=(f[-1, ...] if self.lattice.e[i, 0] > 0 else f[0, ...]).contiguous(),
                      dst=self.next if self.lattice.e[i, 0] > 0 else self.prev)

        return f[1:-2, ...]
