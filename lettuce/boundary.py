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
from lettuce import (LettuceException, torch_gradient)


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

    def __init__(self, lattice, direction):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        direction = np.array(direction)
        self.lattice = lattice

        #select velocities to be bounced (the ones pointing in "direction")
        self.velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, direction) > 1 - 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
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

    def __call__(self, f):
        u = self.lattice.u(f)
        u_w = u[[slice(None)] + self.index] + 0.5 * (u[[slice(None)] + self.index] - u[[slice(None)] + self.neighbor])
        f[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = (
            - f[[self.velocities] + self.index] + self.w * self.lattice.rho(f)[[slice(None)] + self.index] *
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

class GradBoundary:

    def __init__(self, mask, lattice, tau):
        self.relaxation_parameter_lu = tau

        self.obstacle = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.output_force = False
        self.force = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))
        """make masks for fs to be bounced / not streamed by going over all obstacle points and 
        following all e_i's to find neighboring points and which of their fs point towards the obstacle 
        (fs pointing to obstacle are added to no_stream_mask, fs pointing away are added to bouncedFs)"""
        if lattice.D == 2:
            x, y = mask.shape
            self.mask = np.zeros((lattice.Q, x, y), dtype=bool)
            a, b = np.where(mask)
            for p in range(0, len(a)):
                for i in range(0, lattice.Q):
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1]]:
                            self.mask[i, a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if lattice.D == 3:
            x, y, z = mask.shape
            self.mask = np.zeros((lattice.Q, x, y, z), dtype=bool)
            a, b, c = np.where(mask)
            for p in range(0, len(a)):
                for i in range(0, lattice.Q):
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1], c[p] + lattice.stencil.e[i, 2]]:
                            self.mask[i, a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1], c[p] + lattice.stencil.e[i, 2]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        self.mask = self.lattice.convert_to_tensor(self.mask)

    def kronecker(self, A, B):
        return torch.einsum("ab,cd->acbd", A, B).view(A.size(0) * B.size(0), A.size(1) * B.size(1))

    def __call__(self, f):
        #f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        #if self.output_force:
        #    tmp = torch.where(self.mask, f, torch.zeros_like(f))
        #    tmp = torch.einsum("i..., id -> d...", tmp, self.lattice.e[self.lattice.stencil.opposite])
        #    for _ in range(0, self.lattice.D):
        #        tmp = torch.sum(tmp, dim=1)
        #    self.force = 2 * tmp
            # self.force = 1 ** self.lattice.D * 2 * torch.einsum('ixy, id -> d', sum, self.lattice.e) / 1.0
        return f

    def postStreamBoundary(self, f_old, f):
        rho = self.lattice.rho(f_old)
        u = self.lattice.u(f_old)
        j = self.lattice.j(f_old)

        for point in u:
            PIeq = self.kronecker(j[point], u[point]) + rho[point] * self.lattice.cs ** 2 * torch.eye(self.lattice.D)
            PIneq = - rho[point] * self.relaxation_parameter_lu * self.lattice.cs**2 * (torch_gradient(u, 1)  + torch.transpose(torch_gradient(u, 1), 0, 1))
            klammer1 = (PIeq + PIneq - rho[point] * self.lattice.cs**2 * torch.eye(self.lattice.D))
            for i in range(1, self.lattice.Q):
                klammer2 = self.kronecker(self.lattice.e[i], self.lattice.e[i]) - self.lattice.cs**2 * torch.eye(self.lattice.D)
                f[i, point] = self.lattice.w[i] * (rho[point] + rho[point] / self.lattice.cs**2 * torch.einsum('d, d -> ', u[point], self.lattice.e[i]) + 1/(2*self.lattice.cs**4) * torch.einsum('..., ... -> ', klammer1, klammer2))
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.obstacle.shape == f_shape[1:]
        return self.obstacle | self.mask



