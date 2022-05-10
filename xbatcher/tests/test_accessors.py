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


def test_torch_to_tensor(sample_ds_3d):
    torch = pytest.importorskip('torch')

    da = sample_ds_3d['foo']
    t = da.torch.to_tensor()
    assert isinstance(t, torch.Tensor)
    assert t.names == (None, None, None)
    assert t.shape == da.shape
    np.testing.assert_array_equal(t, da.values)


def test_torch_to_named_tensor(sample_ds_3d):
    torch = pytest.importorskip('torch')

    da = sample_ds_3d['foo']
    t = da.torch.to_named_tensor()
    assert isinstance(t, torch.Tensor)
    assert t.names == da.dims
    assert t.shape == da.shape
    np.testing.assert_array_equal(t, da.values)
