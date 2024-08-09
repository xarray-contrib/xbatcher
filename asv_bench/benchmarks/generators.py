import numpy as np
import pandas as pd
import xarray as xr

from xbatcher import BatchGenerator

from . import parameterized, randn

nx = 250
ny = 50
nt = 10

randn_xyt = randn((nx, ny, nt), frac_nan=0.1)


class Generator:
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

    @parameterized(['preload_batch'], ([True, False]))
    def time_batch_preload(self, preload_batch):
        """
        Construct a generator on a chunked DataSet with and without preloading
        batches.
        """
        ds_dask = self.ds.chunk({'t': 2})
        BatchGenerator(ds_dask, input_dims={'t': 2}, preload_batch=preload_batch)

    @parameterized(
        ['input_dims'],
        ([{'x': 10}, {'x': 10, 'y': 5}, {'x': 10, 'y': 5, 't': 2}],),
    )
    def time_input_dims(self, input_dims):
        """
        Benchmark simple batch generation case.
        """
        BatchGenerator(
            self.ds,
            input_dims=input_dims,
        )

    def time_input_dims_and_input_overlap(self):
        """
        Benchmark simple batch generation case.
        """
        BatchGenerator(
            self.ds, input_dims={'x': 10, 'y': 10}, input_overlap={'x': 5, 'y': 5}
        )

    @parameterized(['concat_input_dims'], (['True', 'False']))
    def time_input_dims_and_concat_input_dims(self, concat_input_dims):
        """
        Benchmark concat_input_dims
        """
        BatchGenerator(
            self.ds, input_dims={'x': 10, 'y': 5}, concat_input_dims=concat_input_dims
        )

    @parameterized(
        ['input_dims', 'batch_dims'],
        ([{'x': 10}, {'x': 10, 'y': 5}],),
    )
    def time_input_dims_and_batch_dims(self, input_dims):
        """
        Benchmark batch generator with input_dims and batch_dims.
        """
        BatchGenerator(self.ds, input_dims=input_dims, batch_dims={'t': 2})

    @parameterized(
        ['concat_input_dims'],
        ([True, False]),
    )
    def time_input_dims_batch_dims_and_concat_input_dims(self, concat_input_dims):
        """
        Construct a generator on a DataSet with and without concatenating
        chunks specified by ``input_dims`` into the batch dimension.
        """
        BatchGenerator(
            self.ds,
            input_dims={'x': 10, 'y': 5},
            batch_dims={'x': 20, 'y': 10},
            concat_input_dims=concat_input_dims,
        )

    @parameterized(
        ['concat_input_dims'],
        ([True, False]),
    )
    def time_input_dims_input_overlap_and_concat_input_dims(self, concat_input_dims):
        """
        Construct a generator on a DataSet with and without concatenating
        chunks specified by ``input_dims`` into the batch dimension.
        """
        BatchGenerator(
            self.ds,
            input_dims={'x': 10, 'y': 10},
            input_overlap={'x': 5, 'y': 5},
            concat_input_dims=concat_input_dims,
        )
