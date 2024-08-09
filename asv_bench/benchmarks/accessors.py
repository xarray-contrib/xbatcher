import numpy as np
import pandas as pd
import xarray as xr

import xbatcher  # noqa: F401

from . import parameterized, randn

nx = 250
ny = 50
nt = 10

randn_xyt = randn((nx, ny, nt), frac_nan=0.1)


class Accessor:
    def setup(self, *args, **kwargs):
        self.ds = xr.Dataset(
            {
                'var1': (('x', 'y', 't'), randn_xyt),
            },
            coords={
                'x': np.arange(nx),
                'y': np.linspace(0, 1, ny),
                't': pd.date_range('1970-01-01', periods=nt, freq='D'),
            },
        )

    @parameterized(
        ['input_dims'],
        ([{'x': 10}, {'x': 10, 'y': 5}, {'x': 10, 'y': 5, 't': 2}],),
    )
    def time_input_dims(self, input_dims):
        """
        Benchmark simple batch generation case using xarray accessor
        Equivalent to subset of ``time_batch_input()``.
        """
        bg = self.ds.batch.generator(input_dims=input_dims)
        bg[0]
