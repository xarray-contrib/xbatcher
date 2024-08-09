import numpy as np
import pytest
import xarray as xr

import xbatcher  # noqa: F401
from xbatcher import BatchGenerator


@pytest.fixture(scope='module')
def sample_ds_3d():
    shape = (10, 50, 100)
    ds = xr.Dataset(
        {
            'foo': (['time', 'y', 'x'], np.random.rand(*shape)),
            'bar': (['time', 'y', 'x'], np.random.randint(0, 10, shape)),
        },
        {
            'x': (['x'], np.arange(shape[-1])),
            'y': (['y'], np.arange(shape[-2])),
        },
    )
    return ds


@pytest.fixture(scope='module')
def sample_dataArray():
    return xr.DataArray(np.zeros((2, 4), dtype='i4'), dims=('x', 'y'), name='foo')


@pytest.fixture(scope='module')
def sample_Dataset():
    return xr.Dataset(
        {
            'x': xr.DataArray(np.arange(10), dims='x'),
            'foo': xr.DataArray(np.ones(10, dtype='float'), dims='x'),
        }
    )


def test_as_xarray_dataarray(sample_dataArray, sample_Dataset):
    assert isinstance(
        xbatcher.accessors._as_xarray_dataarray(sample_dataArray), xr.DataArray
    )
    assert isinstance(
        xbatcher.accessors._as_xarray_dataarray(sample_Dataset), xr.DataArray
    )


def test_batch_accessor_ds(sample_ds_3d):
    bg_class = BatchGenerator(sample_ds_3d, input_dims={'x': 5})
    bg_acc = sample_ds_3d.batch.generator(input_dims={'x': 5})
    assert isinstance(bg_acc, BatchGenerator)
    for batch_class, batch_acc in zip(bg_class, bg_acc):
        assert isinstance(batch_acc, xr.Dataset)
        assert batch_class.equals(batch_acc)


def test_batch_accessor_da(sample_ds_3d):
    sample_da = sample_ds_3d['foo']
    bg_class = BatchGenerator(sample_da, input_dims={'x': 5})
    bg_acc = sample_da.batch.generator(input_dims={'x': 5})
    assert isinstance(bg_acc, BatchGenerator)
    for batch_class, batch_acc in zip(bg_class, bg_acc):
        assert batch_class.equals(batch_acc)


@pytest.mark.parametrize(
    'foo_var',
    [
        'foo',  # xr.DataArray
        ['foo'],  # xr.Dataset
    ],
)
def test_tf_to_tensor(sample_ds_3d, foo_var):
    tf = pytest.importorskip('tensorflow')

    foo = sample_ds_3d[foo_var]
    t = foo.tf.to_tensor()
    assert isinstance(t, tf.Tensor)
    assert t.shape == tuple(foo.sizes.values())

    foo_array = foo.to_array().squeeze() if hasattr(foo, 'to_array') else foo
    np.testing.assert_array_equal(t, foo_array.values)


@pytest.mark.parametrize(
    'foo_var',
    [
        'foo',  # xr.DataArray
        ['foo'],  # xr.Dataset
    ],
)
def test_torch_to_tensor(sample_ds_3d, foo_var):
    torch = pytest.importorskip('torch')

    foo = sample_ds_3d[foo_var]
    t = foo.torch.to_tensor()
    assert isinstance(t, torch.Tensor)
    assert t.names == (None, None, None)
    assert t.shape == tuple(foo.sizes.values())

    foo_array = foo.to_array().squeeze() if hasattr(foo, 'to_array') else foo
    np.testing.assert_array_equal(t, foo_array.values)


@pytest.mark.parametrize(
    'foo_var',
    [
        'foo',  # xr.DataArray
        ['foo'],  # xr.Dataset
    ],
)
def test_torch_to_named_tensor(sample_ds_3d, foo_var):
    torch = pytest.importorskip('torch')

    foo = sample_ds_3d[foo_var]
    t = foo.torch.to_named_tensor()
    assert isinstance(t, torch.Tensor)
    assert t.names == tuple(foo.dims)
    assert t.shape == tuple(foo.sizes.values())

    foo_array = foo.to_array().squeeze() if hasattr(foo, 'to_array') else foo
    np.testing.assert_array_equal(t, foo_array.values)
