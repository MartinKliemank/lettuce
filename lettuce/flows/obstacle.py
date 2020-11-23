import numpy as np
from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, AntiBounceBackOutlet, ZeroGradientOutlet
from lettuce.grid import  RegularGrid


class Obstacle2D(object):
    """
    Flow class to simulate the flow around an object (mask) in 2D.
    It consists off one inflow (equilibrium boundary)
    and one outflow (anti-bounce-back-boundary), leading to a flow in positive x direction.

    Parameters
    ----------
    resolution_x : int
        Grid resolution in streamwise direction.
    resolution_y : int
        Grid resolution in spanwise direction.
    char_length_lu : float
        The characteristic length in lattice units; usually the number of grid points for the obstacle in flow direction

    Attributes
    ----------
    mask : np.array with dtype = np.bool
        Boolean mask to define the obstacle. The shape of this object is the shape of the grid.
        Initially set to zero (no obstacle).

    Examples
    --------
    Initialization of flow around a cylinder:

    >>> from lettuce import Lattice, D2Q9
    >>> flow = Obstacle2D(
    >>>     resolution_x=101,
    >>>     resolution_y=51,
    >>>     reynolds_number=100,
    >>>     mach_number=0.1,
    >>>     lattice=lattice,
    >>>     char_length_lu=10
    >>> )
    >>> x, y = flow.grid
    >>> x = flow.units.convert_length_to_lu(x)
    >>> y = flow.units.convert_length_to_lu(y)
    >>> condition = np.sqrt((x-25)**2+(y-25)**2) < 5.0001
    >>> flow.mask[np.where(condition)] = 1
   """
    def __init__(self, resolution_x, resolution_y, reynolds_number, mach_number, lattice, char_length_lu, rank=0, size=1):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y), dtype=np.bool)
        self.grid = RegularGrid([resolution_x, resolution_y], self.units.characteristic_length_lu,
                                self.units.characteristic_length_pu, endpoint=False, rank=rank, size=size)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == (self.resolution_x, self.resolution_y)
        self._mask = m.astype(np.bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u_char = np.array([self.units.characteristic_velocity_pu, 0.0])[..., None, None]
        u = (1 - self.grid.select(self.mask.astype(np.float))) * u_char
        return p, u

    #@property
    #def grid(self):
    #    x = np.linspace(0, self.resolution_x / self.units.characteristic_length_lu, num=self.resolution_x, endpoint=False)
    #    y = np.linspace(0, self.resolution_y / self.units.characteristic_length_lu, num=self.resolution_y, endpoint=False)
     #   return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid.global_grid()
        return [
            EquilibriumBoundaryPU(
                np.abs(x) < 1e-6, self.units.lattice, self.units,
                np.array([self.units.characteristic_velocity_pu, 0])
            ),
            AntiBounceBackOutlet(self.units.lattice, [1, 0]),
            BounceBackBoundary(self.mask, self.units.lattice)
        ]


class Obstacle3D(object):
    """Flow class to simulate the flow around an object (mask) in 3D.
    See documentation for :class:`~Obstacle2D` for details.
    """
    def __init__(self, resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu, rank=0, size=1):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y, self.resolution_z), dtype=np.bool)
        self.grid = RegularGrid([resolution_x, resolution_y, resolution_z], self.units.characteristic_length_lu,
                                self.units.characteristic_length_pu, endpoint=False, rank=rank, size=size)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == (self.resolution_x, self.resolution_y, self.resolution_z)
        self._mask = m.astype(np.bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u_char = np.array([self.units.characteristic_velocity_pu, 0.0, 0.0])[..., None, None, None]
        u = (1 - self.grid.select(self.mask.astype(np.float))) * u_char
        return p, u

    #@property
    #def grid(self):
     #   x = np.linspace(0, self.resolution_x / self.units.characteristic_length_lu, num=self.resolution_x, endpoint=False)
     #   y = np.linspace(0, self.resolution_y / self.units.characteristic_length_lu, num=self.resolution_y, endpoint=False)
     #   z = np.linspace(0, self.resolution_z / self.units.characteristic_length_lu, num=self.resolution_z, endpoint=False)
     #   return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        x, y, z = self.grid.global_grid()
        return [EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units,
                                      np.array([self.units.characteristic_velocity_pu, 0, 0])),
                AntiBounceBackOutlet(self.units.lattice, [1, 0, 0]),
                BounceBackBoundary(self.mask, self.units.lattice)]

class House3D(object):
    """Flow class to simulate the flow around an object (mask) in 3D.
    See documentation for :class:`~Obstacle2D` for details.
    """
    def __init__(self, resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu, rank=0, size=1):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y, self.resolution_z), dtype=np.bool)
        self.grid = RegularGrid([resolution_x, resolution_y, resolution_z], self.units.characteristic_length_lu,
                                self.units.characteristic_length_pu, endpoint=False, rank=rank, size=size)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == (self.resolution_x, self.resolution_y, self.resolution_z)
        self._mask = m.astype(np.bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u_char = np.array([self.units.characteristic_velocity_pu, 0.0, 0.0])[..., None, None, None]
        u = (1 - self.grid.select(self.mask.astype(np.float))) * u_char
        return p, u

    #@property
    #def grid(self):
     #   x = np.linspace(0, self.resolution_x / self.units.characteristic_length_lu, num=self.resolution_x, endpoint=False)
     #   y = np.linspace(0, self.resolution_y / self.units.characteristic_length_lu, num=self.resolution_y, endpoint=False)
     #   z = np.linspace(0, self.resolution_z / self.units.characteristic_length_lu, num=self.resolution_z, endpoint=False)
     #   return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        x, y, z = self.grid.global_grid()
        return [EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units,
                                      np.array([self.units.characteristic_velocity_pu, 0, 0])),
                ZeroGradientOutlet(self.units.lattice, [0, 1, 0]),
                ZeroGradientOutlet(self.units.lattice, [0, -1, 0]),
                ZeroGradientOutlet(self.units.lattice, [0, 0, 1]),
                AntiBounceBackOutlet(self.units.lattice, [1, 0, 0]),
                BounceBackBoundary(self.mask | (z < 1e-6), self.units.lattice)]

    def house(self, o, eg_length, eg_width, eg_height, roof_length, roof_width, angle):
        o[2] = 0
        angle = angle * np.pi / 180
        inside_mask = np.zeros_like(self.grid[0], dtype=bool)
        inside_mask = np.where(np.logical_and(
            np.logical_and(np.logical_and(o[0] - eg_length / 2 < self.grid[0], self.grid[0] < o[0] + eg_length / 2),
                           np.logical_and(o[1] - eg_width / 2 < self.grid[1], self.grid[1] < o[1] + eg_width / 2)),
                            np.logical_and(o[2] - eg_height <= self.grid[2], self.grid[2] <= o[2] + eg_height)), True, inside_mask)
        inside_mask = np.where(np.logical_and(np.logical_and(
                        np.logical_and(np.logical_and(o[0] - roof_length / 2 < self.grid[0], self.grid[0] < o[0] + roof_length / 2),
                        np.logical_and(o[1] - roof_width / 2 < self.grid[1], self.grid[1] < o[1] + roof_width / 2)),
                        np.logical_and(o[2] + eg_height < self.grid[2],
                        self.grid[2] < o[2] + eg_height + 0.001 + np.tan(angle) * (self.grid[1] - o[1] + roof_width / 2))),
                        self.grid[2] < o[2] + eg_height + 0.001 - np.tan(angle) * (self.grid[1] - o[1] - roof_width / 2)), True, inside_mask)

        """make masks for fs to be bounced / not streamed by going over all obstacle points and 
        following all e_i's to find neighboring points and which of their fs point towards the obstacle 
        (fs pointing to obstacle are added to no_stream_mask, fs pointing away are added to bouncedFs)"""

        x, y, z = inside_mask.shape
        outgoing_mask = np.zeros((self.units.lattice.Q, x, y, z), dtype=bool)
        a, b, c = np.where(inside_mask)
        for p in range(0, len(a)):
            for i in range(0, self.units.lattice.Q):
                try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                    if not inside_mask[a[p] + self.units.lattice.stencil.e[i, 0], b[p] + self.units.lattice.stencil.e[i, 1], c[p] + self.units.lattice.stencil.e[i, 2]]:
                        outgoing_mask[i, a[p] + self.units.lattice.stencil.e[i, 0], b[p] + self.units.lattice.stencil.e[i, 1], c[p] + self.units.lattice.stencil.e[i, 2]] = 1
                except IndexError:
                    pass  # just ignore this iteration since there is no neighbor there
        return inside_mask, outgoing_mask

    def wind_speed_profile(self, steps, u_fric, d, z_0):
        # based on wikipedia article about log wind profile
        return u_fric / 0.4 * np.ln(np.linspace(0, steps, steps) - d / z_0)