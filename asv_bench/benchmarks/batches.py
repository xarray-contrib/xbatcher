import numpy as np
import pandas as pd
import xarray as xr

from xbatcher import BatchGenerator

from . import randn

nx = 250
ny = 50
nt = 10

randn_xyt = randn((nx, ny, nt), frac_nan=0.1)


class Base:
    def setup(self):
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


class NoPreload(Base):
    """
    Get a batch from the generator without computing dask arrays.
    """

    def setup(self):
        super().setup()
        ds_dask = self.ds.chunk({'t': 2})
        self.bg = BatchGenerator(ds_dask, input_dims={'t': 2}, preload_batch=False)

    def time_next_batch(self):
        """
        Get a batch
        """
        next(iter(self.bg))


class OneInputDim(Base):
    """
    Get a batch from the generator with one input_dim specified.
    """

    def setup(self):
        super().setup()
        self.bg = BatchGenerator(self.ds, input_dims={'x': 10})

    def time_next_batch(self):
        """
        Get a batch
        """
        next(iter(self.bg))


class AllInputDim(Base):
    """
    Get a batch from the generator with all dimensions specified in input_dims.
    """

    def setup(self):
        super().setup()
        self.bg = BatchGenerator(self.ds, input_dims={'x': 10, 'y': 10, 't': 5})

    def time_next_batch(self):
        """
        Get a batch
        """
        next(iter(self.bg))


class InputDimInputOverlap(Base):
    """
    Get a batch from the generator using input_dims and input_overlap.
    """

    def setup(self):
        super().setup()
        self.bg = BatchGenerator(
            self.ds, input_dims={'x': 10, 'y': 10}, input_overlap={'x': 5, 'y': 5}
        )

    def time_next_batch(self):
        """
        Get a batch
        """
        next(iter(self.bg))


class InputDimConcat(Base):
    """
    Get a batch from the generator with input_dims and concat_input_dims
    """

    def setup(self):
        super().setup()
        self.bg = BatchGenerator(
            self.ds, input_dims={'x': 10, 'y': 10}, concat_input_dims=True
        )

    def time_next_batch(self):
        """
        Get a batch
        """
        next(iter(self.bg))


class InputDimBatchDim(Base):
    """
    Get a batch from the generator with input_dims and batch_dims
    """

    def setup(self):
        super().setup()
        self.bg = BatchGenerator(
            self.ds, input_dims={'x': 10, 'y': 10}, batch_dims={'t': 2}
        )

    def time_next_batch(self):
        """
        Get a batch
        """
        next(iter(self.bg))


class InputDimBatchDimConcat(Base):
    """
    Get a batch from the generator with input_dims, batch_dims and concat_input_dim
    """

    def setup(self):
        super().setup()
        self.bg = BatchGenerator(
            self.ds,
            input_dims={'x': 5, 'y': 5},
            batch_dims={'x': 10, 'y': 10},
            concat_input_dims=True,
        )

    def time_next_batch(self):
        """
        Get a batch
        """
        next(iter(self.bg))


class InputDimInputOverlapConcat(Base):
    """
    Get a batch from the generator with input_dims, input_overlap and concat_input_dim
    """

    def setup(self):
        super().setup()
        self.bg = BatchGenerator(
            self.ds,
            input_dims={'x': 10, 'y': 10},
            input_overlap={'x': 5, 'y': 5},
            concat_input_dims=True,
        )

    def time_next_batch(self):
        """
        Get a batch
        """
        next(iter(self.bg))
