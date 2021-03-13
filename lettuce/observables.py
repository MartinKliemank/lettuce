"""
Observables.
Each observable is defined as a callable class.
The `__call__` function takes f as an argument and returns a torch tensor.
"""


import torch
import numpy as np
import time
from lettuce.util import torch_gradient
from lettuce.boundary import BounceBackBoundary


__all__ = ["Observable", "MaximumVelocity", "IncompressibleKineticEnergy", "Enstrophy", "EnergySpectrum", "Mass",
           "DragCoefficient", "StepTime", "LocalPressure"]


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
        u = self.lattice.u(f)
        return self.spectrum_from_u(u)

    def spectrum_from_u(self, u):
        u = self.flow.units.convert_velocity_to_pu(u)
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
    def __init__(self, lattice, flow, simulation):
        self.lattice = lattice
        self.flow = flow
        self.boundary = []
        for boundary in simulation._boundaries:
            if isinstance(boundary, BounceBackBoundary):
                boundary.output_force = True
                self.boundary.append(boundary)

    def __call__(self, f):
        rho = torch.mean(self.lattice.rho(f[:, 0, ...]))
        Fw = self.boundary[0].force[0]
        #f = torch.where(self.mask, f, torch.zeros_like(f))
        #f[0, ...] = 0
        #Fw =  self.flow.units.convert_force_to_pu(1**self.lattice.D * self.factor * torch.einsum('ixy, id -> d', [f, self.lattice.e])[0]/1)
        drag_coefficient = Fw / (0.5 * rho * self.flow.units.characteristic_velocity_lu**2 * self.flow.area)
        return drag_coefficient

class StepTime(Observable):
    """Outputs the duration of each time step (or multiple steps depending on reporter interval),
    increases execution time by about 0.33 ms per call (on 7th-gen i5 with 4 threads)"""

    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow
        self.time = time.time()

    def __call__(self, f):
        old_time = self.time
        self.time = time.time()
        return self.lattice.convert_to_tensor(self.time - old_time)

class LocalPressure(Observable):
    """The drag coefficient of obstacle, calculated using momentum exchange method"""
    def __init__(self, lattice, flow, coordinates):
        self.lattice = lattice
        self.flow = flow
        #assert ((coordinates[0] > flow.grid.index.start) and (coordinates[0] < flow.grid.index.stop)), \
        #    Exception(f"Process with rank {flow.grid.rank} can't output pressure from domain of other process (at {coordinates})!")
        self.coordinates = []
        print("Coordinates of p reporter in order:")
        for point in coordinates:
            self.coordinates.append([int(x) for x in list(point)])
            print(point)
        #self.coordinates = self.flow.grid.cconvert_coordinate_global_to_local(self.coordinates)

    def __call__(self, f):
        p = torch.zeros(len(self.coordinates), device=self.lattice.device, dtype=self.lattice.dtype)
        p_global = self.flow.grid.reassemble(self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(f)))
        if self.flow.rank == 0:
            for i, point in enumerate(self.coordinates):
                p[i] = p_global[[slice(None)] + point]
        return p