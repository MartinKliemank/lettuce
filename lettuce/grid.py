import numpy as np
from copy import deepcopy
from lettuce import LettuceException

__all__ = ["RegularGrid"]

class RegularGrid(object):

    def __init__(self, resolution, char_length_lu, char_length_pu, endpoint=False, rank=0, size=1):
        """
        class to construct a regular lattice grid for the simulation
        using the rank and size arguments this can be used to split the simulation domain across several processes

        input parameters:
        resolution: list of up to three values for the resolution in x, y, (z)
        char_length_lu/pu: characteristic length of the flow in lu or pu respectively
        endpoint: True if the end of the domain shall be included in the grid; e.g. if 0 to 2pi is [0, 2pi] instead of [0, 2pi)
        rank: rank of the process constructing the grid
        size: total number of processes

        usable parameters:
        self: returns grid as list of two / three elements, one coordinate each [x, y, z]
        shape: returns shape of one of the grid-array without having to construct it first

        function:
        select:
        Inputs:
        tensor (or numpy array) of the size of the grid (if the input is 4D the last three will be assumed to be the grid coordinates)
        rank (optional)

        Output:
        Tensor with part of the input tensor that belongs to process "rank" (calling process if rank is empty / None)
        """
        self.resolution = resolution
        self.char_length_lu = char_length_lu
        self.char_length_pu = char_length_pu
        self.endpoint = endpoint
        self.rank = rank
        self.size = size
        self.index = slice(int(np.floor(self.resolution[0] * self.rank / self.size)),
                           int(np.floor(self.resolution[0] * (self.rank + 1) / self.size)))
        self.shape = deepcopy(self.__call__()[0].shape)

    def __call__(self):
        print("called grid __call__")
        x = np.linspace(0 + self.index.start * self.char_length_pu / self.char_length_lu,
                        self.index.stop * self.char_length_pu / self.char_length_lu,
                        num=self.index.stop - self.index.start, endpoint=self.endpoint)
        y = np.linspace(0, self.resolution[1] * self.char_length_pu / self.char_length_lu, num=self.resolution[1], endpoint=self.endpoint)
        if len(self.resolution) == 3:
            z = np.linspace(0, self.resolution[2] * self.char_length_pu / self.char_length_lu, num=self.resolution[2], endpoint=self.endpoint)
            return np.meshgrid(x, y, z, indexing='ij')
        else:
            return np.meshgrid(x, y, indexing='ij')

    def select(self, tensor, rank=None):
        """reduce tensor (or numpy-array) to the part associated with process "rank" (calling process if rank is None / empty)"""
        if rank is not None:
            assert rank < self.size, LettuceException(f"Calling RegularGrid.select with "
                                                      f"rank ({rank}) >= size ({self.size}), expected rank < size.")
            assert rank >= 0 and (type(rank) is int or type(rank) is float), \
                LettuceException("Calling RegularGrid.select with wrong "
                                 f"rank = {rank} (type: {type(rank)}), expected rank to be a positive int or float.")
            index = slice(int(np.floor(self.resolution[0] * rank / self.size)),
                           int(np.floor(self.resolution[0] * (rank + 1) / self.size)))
        else:
            index = self.index

        if len(tensor.shape) > len(self.resolution):
            return tensor[:, index, ...]
        else:
            return tensor[index, ...]
