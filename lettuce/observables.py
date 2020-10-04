"""
Observables.
Each observable is defined as a callable class.
The `__call__` function takes f as an argument and returns a torch tensor.
"""


import torch
import numpy as np
from lettuce.util import torch_gradient
from lettuce import BounceBackBoundary, HalfWayBounceBackObject


__all__ = ["Observable", "MaximumVelocity", "IncompressibleKineticEnergy", "Enstrophy", "EnergySpectrum",
           "DragCoefficient"]


class Observable:
    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow

    def __call__(self, f):
        raise NotImplementedError


class MaximumVelocity(Observable):
    """Maximum velocitiy"""
    def __call__(self, f):
        u = self.lattice.u(f)
        return self.flow.units.convert_velocity_to_pu(torch.norm(u, dim=0).max())


class IncompressibleKineticEnergy(Observable):
    """Total kinetic energy of an incompressible flow."""
    def __call__(self, f):
        dx = self.flow.units.convert_length_to_pu(1.0)
        kinE = self.flow.units.convert_incompressible_energy_to_pu(torch.sum(self.lattice.incompressible_energy(f)))
        kinE *= dx ** self.lattice.D
        return kinE


class Enstrophy(Observable):
    """The integral of the vorticity

    Notes
    -----
    The function only works for periodic domains
    """
    def __call__(self, f):
        u0 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[0])
        u1 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[1])
        dx = self.flow.units.convert_length_to_pu(1.0)
        grad_u0 = torch_gradient(u0, dx=dx, order=6)
        grad_u1 = torch_gradient(u1, dx=dx, order=6)
        vorticity = torch.sum((grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0]))
        if self.lattice.D == 3:
            u2 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[2])
            grad_u2 = torch_gradient(u2, dx=dx, order=6)
            vorticity += torch.sum(
                (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])
                + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))
            )
        return vorticity * dx**self.lattice.D


class EnergySpectrum(Observable):
    """The kinetic energy spectrum"""
    def __init__(self, lattice, flow):
        super(EnergySpectrum, self).__init__(lattice, flow)
        self.dx = self.flow.units.convert_length_to_pu(1.0)
        self.dimensions = self.flow.grid[0].shape
        frequencies = [self.lattice.convert_to_tensor(np.fft.fftfreq(dim, d=1 / dim)) for dim in self.dimensions]
        wavenumbers = torch.stack(torch.meshgrid(*frequencies))
        wavenorms = torch.norm(wavenumbers, dim=0)
        self.norm = self.dimensions[0] * np.sqrt(2 * np.pi) / self.dx ** 2 if self.lattice.D == 3 else self.dimensions[0] / self.dx
        self.wavenumbers = torch.arange(int(torch.max(wavenorms)))
        self.wavemask = (
            (wavenorms[..., None] > self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) - 0.5) &
            (wavenorms[..., None] <= self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) + 0.5)
        )

    def __call__(self, f):
        u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
        zeros = torch.zeros(self.dimensions, dtype=self.lattice.dtype, device=self.lattice.device)[..., None]
        uh = (torch.stack([
            torch.fft(torch.cat((u[i][..., None], zeros), self.lattice.D),
                      signal_ndim=self.lattice.D) for i in range(self.lattice.D)]) / self.norm)
        ekin = torch.sum(0.5 * (uh[...,0]**2 + uh[...,1]**2), dim=0)
        ek = ekin[..., None] * self.wavemask.to(dtype=self.lattice.dtype)
        ek = ek.sum(torch.arange(self.lattice.D).tolist())
        return ek


class Mass(Observable):
    """Total mass in lattice units.

    Parameters
    ----------
    no_mass_mask : torch.Tensor
        Boolean mask that defines grid points
        which do not count into the total mass (e.g. bounce-back boundaries).
    """
    def __init__(self, lattice, flow, no_mass_mask=None):
        super(Mass, self).__init__(lattice, flow)
        self.mask = no_mass_mask

    def __call__(self, f):
        mass = f[...,1:-1,1:-1].sum()
        if self.mask is not None:
            mass -= (f*self.mask.to(dtype=torch.float)).sum()
        return mass

class DragCoefficient(Observable):
    """The drag coefficient of obstacle, calculated using momentum exchange method"""
    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow
        for boundary in flow.boundaries:
            if isinstance(boundary, HalfWayBounceBackObject):
                self.mask = boundary.mask
                self.factor = 2
            elif isinstance(boundary, BounceBackBoundary):
                mask = self.lattice.convert_to_numpy(boundary.mask)
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
                                    self.mask[self.lattice.stencil.opposite[i], a[p], b[p]] = 1
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
                                    self.mask[self.lattice.stencil.opposite[i], a[p], b[p], c[p]] = 1
                            except IndexError:
                                pass  # just ignore this iteration since there is no neighbor there

                self.mask = self.lattice.convert_to_tensor(self.mask)
                self.factor = 2

    def __call__(self, f):
        rho = self.lattice.rho(f)[:, 0, 0, 0]
        f = torch.where(self.mask, f, torch.zeros_like(f))
        f[0, ...] = 0
        Fw =  self.flow.units.convert_force_to_pu(1**self.lattice.D * self.factor * torch.einsum('ixyz, id -> d', [f, self.lattice.e])[0]/1)
        drag_coefficient = Fw / (0.5 * self.flow.units.convert_density_to_pu(rho) * self.flow.units.characteristic_velocity_pu**2 * self.flow.area)
        return drag_coefficient
