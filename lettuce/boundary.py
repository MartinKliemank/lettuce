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


__all__ = ["BounceBackBoundary", "AntiBounceBackOutlet", "EquilibriumBoundaryPU", "EquilibriumOutletP",
           "ZeroGradientOutlet", "BounceBackVelocityInlet", "EquilibriumExtrapolationOutlet",
           "NonEquilibriumExtrapolationOutlet", "NonEquilibriumExtrapolationInletU", "ConvectiveBoundaryOutlet",
           "HalfWayBounceBackWall", "HalfWayBounceBackObject", "BounceBackWall", "KineticBoundaryOutlet"]


class DirectionalBoundary:
    """Base class for implementing boundaries that apply along an entire side of the domain, do not make objects of this"""

    def __init__(self, lattice, direction):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        self.direction = np.array(direction)
        self.lattice = lattice

        # dimension in which the boundary applies (don't forget to add 1, when indexing f / u)
        self.dim = np.argwhere(self.direction != 0).item()
        # select velocities pointing out of / into the domain
        self.velocities_out = np.concatenate(np.argwhere(lattice.stencil.e[:, self.dim] == direction[self.dim]))
        self.velocities_in = np.array(lattice.stencil.opposite)[self.velocities_out]

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in self.direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)

    def __call__(self, f):
        pass

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[self.velocities_in] + self.index] = 1
        return no_stream_mask


class ObjectBoundary(object):
    pass

"""
class BounceBackBoundary:
    ""Fullway Bounce-Back Boundary""
    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

    def __call__(self, f, index=...):
        f = torch.where(self.mask[index], f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, grid_shape):
        assert self.mask.shape == grid_shape
        return self.mask
"""

class BounceBackBoundary:
    """Fullway Bounce-Back Boundary"""
    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

        self.output_force = False
        self.force = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))
        if lattice.D == 2:
            x, y = mask.shape
            self.force_mask = np.zeros((lattice.Q, x, y), dtype=bool)
            a, b = np.where(mask)
            for p in range(0, len(a)):
                for i in range(0, lattice.Q):
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1]]:
                            self.force_mask[self.lattice.stencil.opposite[i], a[p], b[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if lattice.D == 3:
            x, y, z = mask.shape
            self.force_mask = np.zeros((lattice.Q, x, y, z), dtype=bool)
            a, b, c = np.where(mask)
            for p in range(0, len(a)):
                for i in range(0, lattice.Q):
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1], c[p] + lattice.stencil.e[i, 2]]:
                            self.force_mask[self.lattice.stencil.opposite[i], a[p], b[p], c[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        self.force_mask = self.lattice.convert_to_tensor(self.force_mask)

    def __call__(self, f, index=...):
        if self.output_force:
            tmp = torch.where(self.force_mask, f, torch.zeros_like(f))
            self.force = 1 ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / 1.0
            #tmp = torch.einsum("i..., id -> d...", tmp, self.lattice.e)
            #for _ in range(0, self.lattice.D):
            #    tmp = torch.sum(tmp, dim=1)
            #self.force = tmp * 2
        f = torch.where(self.mask[index], f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, grid_shape):
        assert self.mask.shape == grid_shape
        return self.mask

class BounceBackWall(DirectionalBoundary):
    #TODO: im streaming ist die no_stream_maske zu groß...
    """Fullway Bounce-Back Boundary"""

    def __init__(self, lattice, direction, grid):
        super().__init__(lattice, direction)
        self.ghost_layer = torch.zeros((lattice.Q,) + grid.shape, dtype=lattice.dtype, device=lattice.device)[[slice(None)] + self.index]
        self.streaming = None
        self.grid = grid
        #if grid.size > 1:
            #self.streaming = DistributedStreaming(lattice, grid.rank, grid.size)
        #else:
            #self.streaming = StandardStreaming(lattice)

    def __call__(self, f):#, f_old):
        if np.sum(self.direction) > 0:
            tmp = torch.cat((f[[slice(None)] + self.index], self.ghost_layer), dim=self.dim + 1)
            pos = 1
        else:
            tmp = torch.cat((self.ghost_layer.unsqueeze(self.dim + 1), f[[slice(None)] + self.index].unsqueeze(self.dim + 1)), dim=self.dim + 1)
            pos = 2
        pad = [0 for _ in range((self.lattice.D -1 + 1) * 2)]
        pad[(self.lattice.D - 1 - self.dim) * 2] = 1
        pad[(self.lattice.D - 1 - self.dim) * 2 + 1] = 1
        tmp = torch.nn.functional.pad(tmp, pad)

        self.grid.reassemble(tmp)
        if self.grid.rank == 0:
            for i in range(self.lattice.Q):
                tmp = self._stream(tmp, i)
        #self.streaming(tmp)
        #if self.grid.rank > 0: # TODO
        #    input = torch.zeros_like(tmp[0, ...])
        #    inp = torch.dist.recv(tensor=input.contiguous(), src=0)
        #elif self.grid.rank == 0:
        #    torch.dist.scatter(to all den ganzen / den jeweiligen Teil)

        index = [slice(None) for _ in range(self.lattice.D + 1)]
        index[self.dim + 1] = pos
        f[[slice(None)] + self.index] = tmp[index]
        #streamen in und aus ghost layer, dann die in ghost layer umdrehen
        self.ghost_layer = self.ghost_layer[self.lattice.stencil.opposite]
        return f

    def _stream(self, f, i):
        return torch.roll(f[i], shifts=tuple(self.lattice.stencil.e[i]), dims=tuple(np.arange(self.lattice.D)))

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

    def __call__(self, f, index=...):
        rho = self.units.convert_pressure_pu_to_density_lu(self.pressure)
        u = self.units.convert_velocity_to_lu(self.velocity)
        feq = self.lattice.equilibrium(rho, u)
        feq = self.lattice.einsum("q,q->q", [feq, torch.ones_like(f)])
        f = torch.where(self.mask[index], feq, f)
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
        self.direction = np.array(direction)
        self.lattice = lattice

        #select velocities to be bounced (the ones pointing in "direction")
        self.velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) > 1 - 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in self.direction:
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
        if len(self.direction) == 3:
            self.dims = 'dc, cxy -> dxy'
            self.w = self.lattice.w[self.velocities].view(1, -1).t().unsqueeze(1)
        if len(self.direction) == 2:
            self.dims = 'dc, cx -> dx'
            self.w = self.lattice.w[self.velocities].view(1, -1).t()
        if len(self.direction) == 1:
            self.dims = 'dc, c -> dc'
            self.w = self.lattice.w[self.velocities]

    def __call__(self, f):
        #TODO ? geht kaputt wenn u_w < 0!!!
        u = self.lattice.u(f)
        u_w = u[[slice(None)] + self.index] + 0.5 * (u[[slice(None)] + self.index] - u[[slice(None)] + self.neighbor])
        f[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = (
            - f[[self.velocities] + self.index] + self.w * self.lattice.rho(f)[[slice(None)] + self.index] *
            (2 + torch.einsum(self.dims, self.lattice.e[self.velocities], u_w) ** 2 / self.lattice.cs ** 4
             - (torch.norm(u_w, dim=0) / self.lattice.cs) ** 2)
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

    def make_no_collision_mask(self, grid_shape):
        no_collision_mask = torch.zeros(size=grid_shape, dtype=torch.bool, device=self.lattice.device)
        no_collision_mask[self.index] = 1
        return no_collision_mask

class EquilibriumExtrapolationOutlet(AntiBounceBackOutlet):
    """Equilibrium outlet with extrapolated pressure and velocity from inside the domain
    """
    def __init__(self, lattice, direction):
        super(EquilibriumExtrapolationOutlet, self).__init__(lattice, direction)

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        rho_w = rho[other] #+ 0.5 * (rho[here] - rho[other])
        u_w = u[other] #+ 0.5 * (u[here] - u[other])
        f[here] = self.lattice.equilibrium(rho_w[...,None], u_w[...,None])[...,0]
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.setdiff1d(np.arange(self.lattice.Q), self.velocities)] + self.index] = 1
        return no_stream_mask

    def make_no_collision_mask(self, grid_shape):
        no_collision_mask = torch.zeros(size=grid_shape, dtype=torch.bool, device=self.lattice.device)
        no_collision_mask[self.index] = 1
        return no_collision_mask

class ZeroGradientOutlet(object):

    def __init__(self, lattice, direction):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        self.direction = np.array(direction)
        self.lattice = lattice

        #select velocities to be replaced (the ones pointing against "direction")
        self.velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) < -1 + 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in self.direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)

    def __call__(self, f):
        f[[self.velocities] + self.index] = f[[self.velocities] + self.neighbor]
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[self.velocities] + self.index] = 1
        return no_stream_mask


class BounceBackVelocityInlet(object):
    """Allows distributions to enter domain with set speed through this boundary.
        Based on page 195 of "The lattice Boltzmann method" (2016 by Krüger et al.)
        Give the side of the domain with the boundary as list [x, y, z] with only one entry nonzero
        [1, 0, 0] for positive x-direction in 3D; [1, 0] for the same in 2D
        [0, -1, 0] is negative y-direction in 3D; [0, -1] for the same in 2D
        """

    def __init__(self, lattice, units, direction, velocity_pu):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        self.direction = np.array(direction)
        self.lattice = lattice
        self.units = units
        self.velocity_lu = units.convert_velocity_to_lu(self.lattice.convert_to_tensor(velocity_pu))

        #select velocities to be bounced (the ones pointing in "direction")
        self.velocities_out = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) > 1 - 1e-6), axis=0)
        # select velocities to be replaced (the ones pointing against "direction")
        self.velocities_in = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) < -1 + 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in self.direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)

    def __call__(self, f):
        rho = self.lattice.rho(f[[slice(None)] + self.neighbor])
        rho_w = torch.mean(rho) #rho[[slice(None)] + self.index] + 0.5 * (rho[[slice(None)] + self.index] - rho[[slice(None)] + self.neighbor])  # extrapolation of rho_w from density at boundary and neighbour node, hopefully better than global average / 1
        list = []
        for _ in rho.shape: list += [1]
        list[0] = len(self.velocities_out)
        f[[self.velocities_in] + self.index] = (
                f[[self.velocities_out] + self.index] - 2 * rho_w * (self.lattice.w[self.velocities_out] * torch.matmul(self.lattice.e[self.velocities_out], self.velocity_lu) / self.lattice.cs ** 2).view(list)
        )
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[self.velocities_in] + self.index] = 1
        return no_stream_mask

class NonEquilibriumExtrapolationOutlet(object):
    """ Zou's boundary condition
    use on post stream populations!!!!!
        """

    def __init__(self, lattice, rho_w, direction):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        self.direction = np.array(direction)
        self.lattice = lattice
        self.rho_w = self.lattice.convert_to_tensor(rho_w)

        # select velocities to be bounced (the ones pointing in "direction")
        self.velocities_out = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) > 1 - 1e-6), axis=0)
        # select velocities to be replaced (the ones pointing against "direction")
        self.velocities_in = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) < -1 + 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in self.direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        u = self.lattice.convert_to_tensor(self.lattice.u(f[other]))
        u_w = u.detach().clone()
        # smart (?) methode to find velocity orthogonal to boundary... doesnt help for parallel velocity though
        u_w[np.argwhere(self.direction != 0).item()] = self.lattice.e[self.velocities_in[0], np.argwhere(self.direction != 0).item()] * (1 - 1 / self.rho_w * (
            torch.sum(f[[np.setdiff1d(np.arange(self.lattice.Q), [self.velocities_in, self.velocities_out])] + self.index]
                      + 2 * f[[self.velocities_out] + self.index], dim=0)))
        rho = self.lattice.convert_to_tensor(self.lattice.rho(f[other]))
        rho_w = self.rho_w * torch.ones_like(rho)
        f[here] = self.lattice.equilibrium(rho_w, u_w) + (f[other] - self.lattice.equilibrium(rho, u))
        return f

   # def make_no_stream_mask(self, f_shape):
   #     no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
   #     no_stream_mask[[self.velocities_in] + self.index] = 1
   #     return no_stream_mask

# der ist gut?^^
class NonEquilibriumExtrapolationInletU(object):
    """ Guo's boundary condition
    use on post stream populations!!!!!
    https://www.researchgate.net/publication/230963379_Non-equilibrium_extrapolation_method_for_velocity_and_boundary_conditions_in_the_lattice_Boltzmann_method
    or LBM book page 189
        """

    def __init__(self, lattice, u_w, direction):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        self.direction = np.array(direction)
        self.lattice = lattice
        self.u_w = self.lattice.convert_to_tensor(u_w)

        # select velocities to be bounced (the ones pointing in "direction")
        self.velocities_out = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) > 1 - 1e-6), axis=0)
        # select velocities to be replaced (the ones pointing against "direction")
        self.velocities_in = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) < -1 + 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in self.direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)
        self.rho_old = 1.0
        if len(self.u_w.shape) > self.lattice.D:
            self.u_w = self.u_w[tuple([slice(None)] + self.index)]

    def __call__(self, f):
        Tc = 10
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        u = self.lattice.convert_to_tensor(self.lattice.u(f[other]))
        rho = self.lattice.convert_to_tensor(self.lattice.rho(f[other]))
        if self.u_w.shape == u.shape:
            u_w = self.u_w
        else:
            list = []
            for _ in u.shape: list += [1]
            list[0] = len(self.u_w)
            u_w = self.u_w.view(list).expand_as(u)
        rho_self = 1 / (1 - u_w[np.argwhere(self.direction != 0).item()] / self.lattice.e[self.velocities_in[0], np.argwhere(self.direction != 0).item()]) * (
            torch.sum(f[[np.setdiff1d(np.arange(self.lattice.Q), [self.velocities_in, self.velocities_out])] + self.index] + 2 * f[[self.velocities_out] + self.index], dim=0))
        # desnity filtering as proposed by https://www.researchgate.net/publication/257389374_Computational_Gas_Dynamics_with_the_Lattice_Boltzmann_Method_Preconditioning_and_Boundary_Conditions
        rho_w = (rho_self + Tc * self.rho_old) / (1+Tc)
        self.rho_old = rho_w
        f[here] = self.lattice.equilibrium(rho_w, u_w) + (f[other] - self.lattice.equilibrium(rho, u))
        return f

    #def make_no_stream_mask(self, f_shape):
    #    no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
    #    no_stream_mask[[self.velocities_in] + self.index] = 1
    #    return no_stream_mask

class ConvectiveBoundaryOutlet(object):
    """convective boundary as described in: https://www.sciencedirect.com/science/article/pii/S0898122112006736#br000045
        """

    def __init__(self, lattice, rho_w, direction):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        self.direction = np.array(direction)
        self.lattice = lattice
        self.rho_w = self.lattice.convert_to_tensor(rho_w)

        # select velocities to be bounced (the ones pointing in "direction")
        self.velocities_out = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) > 1 - 1e-6), axis=0)
        # select velocities to be replaced (the ones pointing against "direction")
        self.velocities_in = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) < -1 + 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        self.neighbor2 = []
        for i in self.direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
                self.neighbor2.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
                self.neighbor2.append(-3)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)
                self.neighbor2.append(2)

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        another = [slice(None)] + self.neighbor2
        u = self.lattice.convert_to_tensor(self.lattice.u(f))
        u_flat = torch.abs(torch.mean(u[[np.argwhere(self.direction != 0).item()] + self.index]))
        du = - u_flat / 2 * (3 * u[here] - 4 * u[other] + u[another])
        f[[self.velocities_in] + self.index] = f[[self.velocities_in] + self.index] + 3 * self.lattice.w[self.velocities_in].unsqueeze(1) * torch.tensordot(self.lattice.e[self.velocities_in], du, dims=1)
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[self.velocities_in] + self.index] = 1
        return no_stream_mask

class HalfWayBounceBackObject:
    """Halfway Bounce-Back Boundary around object mask"""
    def __init__(self, mask, lattice):
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

    def __call__(self, f):
        f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        if self.output_force:
            tmp = torch.where(self.mask, f, torch.zeros_like(f))
            tmp = torch.einsum("i..., id -> d...", tmp, self.lattice.e[self.lattice.stencil.opposite])
            for _ in range(0, self.lattice.D):
                tmp = torch.sum(tmp, dim=1)
            self.force = 2 * tmp
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.obstacle.shape == f_shape[1:]
        return self.obstacle | self.mask

    def make_no_collision_mask(self, f_shape):
        assert self.obstacle.shape == f_shape[1:]
        return self.obstacle

class HalfWayBounceBackWall:
    """Halfway Bounce-Back Boundary on side of the domain (0 thickness)"""
    def __init__(self, direction, lattice):
        self.lattice = lattice
        direction = np.array(direction)

        # select velocities to be bounced (the ones pointing in "direction")
        velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, direction) > 1 - 1e-6), axis=0)
        index = []
        for i in direction:
            if i == 0:
                index.append(slice(None))
            if i == 1:
                index.append(-1)
            if i == -1:
                index.append(0)

        self.bounced = [np.array(self.lattice.stencil.opposite)[velocities]] + index
        self.outgoing = [velocities] + index

    def __call__(self, f):
        f[self.bounced] = f[self.outgoing]
        return f

    def make_no_stream_mask(self, f_shape):
        mask = np.zeros(f_shape, dtype=bool)
        mask[tuple(self.bounced)] = 1
        mask = self.lattice.convert_to_tensor(mask)
        return mask

class KineticBoundaryOutlet(DirectionalBoundary):

    def __init__(self, lattice, direction):
        super().__init__(lattice, direction)
# TODO prodziert komische Welle und zerstört Dichte völlig :x
    # nach https://www.sciencedirect.com/science/article/pii/S0045793017302189#bib0028 formel (11)

    def __call__(self, f):
        u = self.lattice.u(f)
        rho = self.lattice.rho(f)
        u_w = u[[slice(None)] + self.neighbor] # 0 because stationary wall TODO ist das Unfug? was besseres?
        rho_w = torch.ones_like(rho[[0] + self.index]) # is cancelled in the formula ( sum(x) / sum(rho * y) x (rho * u) )
        f_eq = self.lattice.equilibrium(rho_w, u_w)
        f[[self.velocities_in] + self.index] = torch.einsum("..., u... -> u...", torch.sum(f[[self.velocities_out] + self.index], dim=0) / torch.sum(f_eq[self.velocities_in], dim=0), f_eq[self.velocities_in])
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[self.velocities_in] + self.index] = 1
        return no_stream_mask
