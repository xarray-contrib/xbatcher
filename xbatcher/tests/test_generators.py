import xarray as xr
import numpy as np
from xbatcher import BatchGenerator
import pytest


@pytest.fixture(scope='module')
def sample_ds_1d():
    size = 100
    ds = xr.Dataset({'foo': (['x'], np.random.rand(size)),
                     'bar': (['x'], np.random.randint(0, 10, size))},
                    {'x': (['x'], np.arange(size))})
    return ds


# TODO: decide how to handle bsizes like 15 that don't evenly divide the dimension
# Should we enforce that each batch size always has to be the same
@pytest.mark.parametrize("bsize", [5, 10])
def test_batch_1d(sample_ds_1d, bsize):
    bg = BatchGenerator(sample_ds_1d, input_dims={'x': bsize})
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
    bg = BatchGenerator(sample_ds_1d, input_dims={'x': bsize},
                        input_overlap={'x': olap})
    stride = bsize-olap
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims['x'] == bsize
        expected_slice = slice(stride*n, stride*n + bsize)
        ds_batch_expected = sample_ds_1d.isel(x=expected_slice)
        assert ds_batch.equals(ds_batch_expected)


@pytest.fixture(scope='module')
def sample_ds_3d():
    shape = (10, 50, 100)
    ds = xr.Dataset({'foo': (['time', 'y', 'x'], np.random.rand(*shape)),
                     'bar': (['time', 'y', 'x'], np.random.randint(0, 10, shape))},
                    {'x': (['x'], np.arange(shape[-1])),
                     'y': (['y'], np.arange(shape[-2]))})
    return ds


@pytest.mark.parametrize("bsize", [5, 10])
def test_batch_3d_1d_input(sample_ds_3d, bsize):

    # first do the iteration over just one dimension
    bg = BatchGenerator(sample_ds_3d, input_dims={'x': bsize})
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims['x'] == bsize
        # time and y should be collapsed into batch dimension
        assert ds_batch.dims['sample'] == sample_ds_3d.dims['y'] * sample_ds_3d.dims['time']
        expected_slice = slice(bsize*n, bsize*(n+1))
        ds_batch_expected = (sample_ds_3d.isel(x=expected_slice)
                                         .stack(sample=['y', 'time'])
                                         .transpose('sample', 'x'))
        print(ds_batch)
        print(ds_batch_expected)
        assert ds_batch.equals(ds_batch_expected)

@pytest.mark.parametrize("bsize", [5, 10])
def test_batch_3d_2d_input(sample_ds_3d, bsize):
    # now iterate over both
    xbsize = 20
    bg = BatchGenerator(sample_ds_3d, input_dims={'y': bsize, 'x': xbsize})
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims['x'] == xbsize
        assert ds_batch.dims['y'] == bsize
        # TODO? Is it worth it to try to reproduce the internal logic of the
        # generator and verify that the slices are correct?
    assert (n+1)==((sample_ds_3d.dims['x']//xbsize) * (sample_ds_3d.dims['y']//bsize))


@pytest.mark.parametrize("bsize", [5, 10])
def test_batch_3d_2d_input_concat(sample_ds_3d, bsize):
    # now iterate over both
    xbsize = 20
    bg = BatchGenerator(sample_ds_3d, input_dims={'y': bsize, 'x': xbsize},
                        concat_input_dims=True)
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims['x_input'] == xbsize
        assert ds_batch.dims['y_input'] == bsize
        assert ds_batch.dims['sample'] == ((sample_ds_3d.dims['x']//xbsize) *
                                          (sample_ds_3d.dims['y']//bsize) *
                                          sample_ds_3d.dims['time'])
