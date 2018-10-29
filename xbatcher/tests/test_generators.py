import xarray as xr
import numpy as np
from xbatcher import BatchGenerator
import pytest


@pytest.fixture(scope='module')
def sample_ds_1d():
    size=100
    ds = xr.Dataset({'foo': (['x'], np.random.rand(size)),
                     'bar': (['x'], np.random.randint(0, 10, size))},
                    {'x': (['x'], np.arange(size))})
    return ds

# TODO: decide how to handle bsizes like 15 that don't evenly divide the dimension
# Should we enforce that each batch size always has to be the same
@pytest.mark.parametrize("bsize", [5, 10])
def test_batch_1d(sample_ds_1d, bsize):
    bg = BatchGenerator(sample_ds_1d, batch_sizes={'x': bsize})
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        # TODO: maybe relax this? see comment above
        assert ds_batch.dims['x'] == bsize
        expected_slice = slice(bsize*n, bsize*(n+1))
        ds_batch_expected = sample_ds_1d.isel(x=expected_slice)
        assert ds_batch.equals(ds_batch_expected)

@pytest.mark.parametrize("olap", [1, 4])
def test_batch_1d_overlap(sample_ds_1d, olap):
    bsize = 10
    bg = BatchGenerator(sample_ds_1d, batch_sizes={'x': bsize},
                        overlap={'x': olap})
    stride = bsize-olap
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims['x'] == bsize
        expected_slice = slice(stride*n, stride*n + bsize)
        ds_batch_expected = sample_ds_1d.isel(x=expected_slice)
        assert ds_batch.equals(ds_batch_expected)
