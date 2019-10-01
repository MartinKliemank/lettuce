"""
Input/output routines.

TODO: Logging
TODO: VTK field/o
"""

import sys
import logging
import numpy as np
import torch
#from pyevtk.hl import *
import pyevtk.hl as vtk
import pyevtk.vtk as VTKGroup

from matplotlib import pyplot as plt

def write_image(filename, array2d):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.tight_layout()
    ax.imshow(array2d)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(filename)

def write_png(filename, array2d):
    pass


#def write_vtk(filename, res, ux, uy, p, t):
def write_vtk(filename, point_dict, id):

    vtk.gridToVTK("/Users/mariobedrunka/Documents/10_science/10_lattice_boltzmann/10_simulation/10_lettuce/data/" + filename + "_" + str(id),
                  np.arange(0, point_dict["p"].shape[0]),
                  np.arange(0, point_dict["p"].shape[1]),
                  np.arange(0, point_dict["p"].shape[2]),
                  pointData=point_dict)

class VTKReporter:
    """General VTK Reporter for velocity and pressure"""
    def __init__(self, lattice, flow, filename, interval):
        self.lattice = lattice
        self.flow = flow
        self.interval = interval
        self.filename = filename
        self.point_dict = dict()

    def __call__(self, i, t, f):
        if t % self.interval == 0:
            t = self.flow.units.convert_time_to_pu(t)
            u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(f))
            self.point_dict["p"] = self.lattice.convert_to_numpy(p[0, ..., None])
            for d in range(self.lattice.D):
                self.point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ..., None])
            write_vtk(self.filename, self.point_dict, i)

class ErrorReporter:
    """Reports numerical errors with respect to analytic solution."""
    def __init__(self, lattice, flow, interval=1, out=sys.stdout):
        assert hasattr(flow, "analytic_solution")
        self.lattice = lattice
        self.flow = flow
        self.interval = interval
        self.out = [] if out is None else out
        if not isinstance(self.out, list):
            print("#error_u         error_p", file=self.out)

    def __call__(self, i, t, f):
        if t % self.interval == 0:
            t = self.flow.units.convert_time_to_pu(t)
            pref, uref = self.flow.analytic_solution(self.flow.grid, t=t)
            pref = self.lattice.convert_to_tensor(pref)
            uref = self.lattice.convert_to_tensor(uref)
            u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(f))

            resolution = torch.pow(torch.prod(self.lattice.convert_to_tensor(p.size())),1/self.lattice.D)

            err_u = torch.norm(u-uref)/resolution
            err_p = torch.norm(p-pref)/resolution

            if isinstance(self.out, list):
                self.out.append([err_u.item(), err_p.item()])
            else:
                print(err_u.item(), err_p.item(), file=self.out)

