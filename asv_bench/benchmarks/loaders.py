import numpy as np
import torch
import xarray as xr

from xbatcher import BatchGenerator
from xbatcher.loaders.torch import IterableDataset, MapDataset

from . import randn

nx = 250
ny = 50

randn_xy = randn((nx, ny), frac_nan=0.1)
randn_y = randn((ny), frac_nan=0.1)


class TorchLoader:
    def setup(self, *args, **kwargs):
        self.ds = xr.Dataset(
            {
                'var1': (('x', 'y'), randn_xy),
                'var2': (('y'), randn_y),
            },
            coords={
                'x': np.arange(nx),
                'y': np.linspace(0, 1, ny),
            },
        )
        self.x_gen = BatchGenerator(self.ds['var1'], {'y': 10})
        self.y_gen = BatchGenerator(self.ds['var2'], {'y': 10})

    def time_map_dataset(self):
        """
        Benchmark MapDataset integration with torch DataLoader.
        """
        dataset = MapDataset(self.x_gen, self.y_gen)
        loader = torch.utils.data.DataLoader(dataset)
        next(iter(loader))

    def time_iterable_dataset(self):
        """
        Benchmark IterableDataset integration with torch DataLoader.
        """
        dataset = IterableDataset(self.x_gen, self.y_gen)
        loader = torch.utils.data.DataLoader(dataset)
        next(iter(loader))
