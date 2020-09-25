"""
Boundary Conditions.

The `__call__` function of a boundary defines its application to the distribution functions.

Boundary conditions can define a mask (a boolean numpy array)
that specifies the grid points on which the boundary
condition operates.

Boundary classes can define two functions `make_no_stream_mask` and `make_no_collision_mask`
that prevent streaming and collisions on the boundary nodes.

The no-stream mask has the same dimensions as the distribution functions (Q, x, y, (z)) .
The no-collision mask has the same dimensions as the grid (x, y, (z)).

"""

import torch
import numpy as np
from lettuce import (LettuceException)


__all__ = ["BounceBackBoundary", "AntiBounceBackOutlet", "EquilibriumBoundaryPU", "EquilibriumOutletP"]


class BounceBackBoundary:
    """Fullway Bounce-Back Boundary"""
    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

    def __call__(self, f):
        f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask


class EquilibriumBoundaryPU:
    """Sets distributions on this boundary to equilibrium with predefined velocity and pressure.
    Note that this behavior is generally not compatible with the Navier-Stokes equations.
    This boundary condition should only be used if no better options are available.
    """
    def __init__(self, mask, lattice, units, velocity, pressure=0):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.units = units
        self.velocity = lattice.convert_to_tensor(velocity)
        self.pressure = lattice.convert_to_tensor(pressure)

    def __call__(self, f):
        rho = self.units.convert_pressure_pu_to_density_lu(self.pressure)
        u = self.units.convert_velocity_to_lu(self.velocity)
        feq = self.lattice.equilibrium(rho, u)
        feq = self.lattice.einsum("q,q->q", [feq, torch.ones_like(f)])
        f = torch.where(self.mask, feq, f)
        return f


class AntiBounceBackOutlet:
    """Allows distributions to leave domain unobstructed through this boundary.
        Based on equations from page 195 of "The lattice Boltzmann method" (2016 by Krüger et al.)
        Give the side of the domain with the boundary as list [x, y, z] with only one entry nonzero
        [1, 0, 0] for positive x-direction in 3D; [1, 0] for the same in 2D
        [0, -1, 0] is negative y-direction in 3D; [0, -1] for the same in 2D
        """

    def __init__(self, lattice, direction, corners=False):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        self.direction = np.array(direction)
        self.lattice = lattice

        #select velocities to be bounced (the ones pointing in "direction")
        self.velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, direction) > 1 - 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        self.indexInner = []
        self.corners = corners
        for i in direction:
            if i == 0:
                self.index.append(slice(None))
                self.indexInner.append(slice(1,-1))
                self.neighbor.append(slice(None))
            if i == 1:
                self.indexInner.append(-1)
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.indexInner.append(0)
                self.index.append(0)
                self.neighbor.append(1)
        # construct indices for einsum and get w in proper shape for the calculation in each dimension
        if len(direction) == 3:
            self.dims = 'dc, cxy -> dxy'
            self.w = self.lattice.w[self.velocities].view(1, -1).t().unsqueeze(1)
        if len(direction) == 2:
            self.dims = 'dc, cx -> dx'
            self.w = self.lattice.w[self.velocities].view(1, -1).t()
        if len(direction) == 1:
            self.dims = 'dc, c -> dc'
            self.w = self.lattice.w[self.velocities]

        #--------------2D only---------------------
        self.corner1 = [0 if x == slice(None) else x for x in self.index]
        self.corner2 = [-1 if x == slice(None) else x for x in self.index]
        #self.cornerUw1 = [x for x in self.corner1]
        #self.cornerUw2 = [x for x in self.corner2]
        #blab = int(np.where(np.array(self.corner1) == np.array(self.corner2))[0])
        self.cornerUw1 = [slice(None), 0]
        self.cornerUw2 = [slice(None), -1]
        #self.cornerNeighbor1 = [1 if x == slice(None) else x for x in self.neighbor]
        #self.cornerNeighbor2 = [-2 if x == slice(None) else x for x in self.neighbor]

    def __call__(self, f):
        u = self.lattice.u(f)
        u_w = u[[slice(None)] + self.index] + 0.5 * (u[[slice(None)] + self.index] - u[[slice(None)] + self.neighbor])

        #orthogonal
        i = self.velocities[torch.where(self.lattice.e[self.velocities] == 0)[0]]
        f[[[np.array(self.lattice.stencil.opposite)[i]]] + self.corner1] = (
                - f[[[i]] + self.corner1] + self.lattice.w[i] * self.lattice.rho(f)[[slice(None)] + self.corner1] *
                (2 + torch.einsum('c, c ->', self.lattice.e[i], u_w[self.cornerUw1]) ** 2 / self.lattice.cs ** 4
                - (torch.norm(u_w[self.cornerUw1], dim=0) / self.lattice.cs) ** 2)
        )
        f[[[self.lattice.stencil.opposite[i]]] + self.corner2] = (
                - f[[[i]] + self.corner2] + self.lattice.w[i] * self.lattice.rho(f)[[slice(None)] + self.corner2] *
                (2 + torch.einsum('c, c ->', self.lattice.e[i], u_w[self.cornerUw2]) ** 2 / self.lattice.cs ** 4
                - (torch.norm(u_w[self.cornerUw2], dim=0) / self.lattice.cs) ** 2)
        )
        if self.corners:
            #diagonal nach innen
            i1 = int(self.velocities[[self.lattice.stencil.e[self.velocities][:, [np.where(self.direction == 0)[0][0]]] == -1][0].squeeze()])
            i2 = int(self.velocities[[self.lattice.stencil.e[self.velocities][:, [np.where(self.direction == 0)[0][0]]] == 1][0].squeeze()])
            f[[[self.lattice.stencil.opposite[i1]]] + self.corner1] = (
                    - f[[[i1]] + self.corner1] + self.lattice.w[i1] * self.lattice.rho(f)[[slice(None)] + self.corner1] *
                    (2 + torch.einsum('c, c ->', self.lattice.e[i1], u_w[self.cornerUw1]) ** 2 / self.lattice.cs ** 4
                     - (torch.norm(u_w[self.cornerUw1], dim=0) / self.lattice.cs) ** 2)
            )
            f[[[self.lattice.stencil.opposite[i2]]] + self.corner2] = (
                    - f[[[i2]] + self.corner2] + self.lattice.w[i2] * self.lattice.rho(f)[[slice(None)] + self.corner2] *
                    (2 + torch.einsum('c, c ->', self.lattice.e[i2], u_w[self.cornerUw2]) ** 2 / self.lattice.cs ** 4
                     - (torch.norm(u_w[self.cornerUw2], dim=0) / self.lattice.cs) ** 2)
            )
        #    u_w[:, 0] = u[[slice(None)] + self.corner1] + np.sqrt(0.5) * (u[[slice(None)] + self.corner1] - u[[slice(None)] + self.cornerNeighbor1])
        #    u_w[:, -1] = u[[slice(None)] + self.corner2] + np.sqrt(0.5) * (u[[slice(None)] + self.corner2] - u[[slice(None)] + self.cornerNeighbor2])
        u_w = u_w[:, 1:-1]
        f[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.indexInner] = (
            - f[[self.velocities] + self.indexInner] + self.w * self.lattice.rho(f)[[slice(None)] + self.indexInner] *
            (2 + torch.einsum(self.dims, self.lattice.e[self.velocities], u_w) ** 2 / self.lattice.cs ** 4
             - (torch.norm(u_w,dim=0) / self.lattice.cs) ** 2)
        )
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = 1
        return no_stream_mask

    # not 100% sure about this. But collisions seem to stabilize the boundary.
    #def make_no_collision_mask(self, f_shape):
    #    no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
    #    no_collision_mask[self.index] = 1
    #    return no_collision_mask


class EquilibriumOutletP(AntiBounceBackOutlet):
    """Equilibrium outlet with constant pressure.
    """
    def __init__(self, lattice, direction, rho0=1.0):
        super(EquilibriumOutletP, self).__init__(lattice, direction)
        self.rho0 = rho0

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        rho_w = self.rho0 * torch.ones_like(rho[here])
        u_w = u[other]
        f[here] = self.lattice.equilibrium(rho_w[...,None], u_w[...,None])[...,0]
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.setdiff1d(np.arange(self.lattice.Q), self.velocities)] + self.index] = 1
        return no_stream_mask

    def make_no_collision_mask(self, f_shape):
        no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
        no_collision_mask[self.index] = 1
        return no_collision_mask
