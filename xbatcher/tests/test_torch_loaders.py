from importlib import reload

import numpy as np
import pytest
import xarray as xr

from xbatcher import BatchGenerator
from xbatcher.loaders.torch import IterableDataset, MapDataset, to_tensor

torch = pytest.importorskip('torch')


def test_import_torch_failure(monkeypatch):
    import sys

    import xbatcher.loaders

    monkeypatch.setitem(sys.modules, 'torch', None)

    with pytest.raises(ImportError) as excinfo:
        reload(xbatcher.loaders.torch)

    assert 'install PyTorch to proceed' in str(excinfo.value)


def test_import_dask_failure(monkeypatch):
    import sys

    import xbatcher.loaders

    monkeypatch.setitem(sys.modules, 'dask', None)
    reload(xbatcher.loaders.torch)

    assert xbatcher.loaders.torch.dask is None


@pytest.fixture(scope='module', params=[True, False])
def ds_xy(request):
    n_samples = 100
    n_features = 5
    ds = xr.Dataset(
        {
            'x': (
                ['sample', 'feature'],
                np.random.random((n_samples, n_features)),
            ),
            'y': (['sample'], np.random.random(n_samples)),
        },
    )

    if request.param:
        ds = ds.chunk({'sample': 10})

    return ds


@pytest.mark.parametrize('x_var', ['x', ['x']])
def test_map_dataset_without_y(ds_xy, x_var) -> None:
    x = ds_xy[x_var]

    x_gen = BatchGenerator(x, {'sample': 10})

    dataset = MapDataset(x_gen)

    # test __getitem__
    x_batch = dataset[0]
    assert x_batch.shape == (10, 5)  # type: ignore[union-attr]
    assert isinstance(x_batch, torch.Tensor)

    idx = torch.tensor([0])
    x_batch = dataset[idx]
    assert x_batch.shape == (10, 5)
    assert isinstance(x_batch, torch.Tensor)

    with pytest.raises(NotImplementedError):
        idx = torch.tensor([0, 1])
        x_batch = dataset[idx]

    # test __len__
    assert len(dataset) == len(x_gen)

    # test integration with torch DataLoader
    loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    for x_batch in loader:
        assert x_batch.shape == (10, 5)  # type: ignore[union-attr]
        assert isinstance(x_batch, torch.Tensor)

    # Check that array shape of last item in generator is same as the batch image
    assert tuple(x_gen[-1].sizes.values()) == x_batch.shape  # type: ignore[union-attr]
    # Check that array values from last item in generator and batch are the same
    gen_array = (
        x_gen[-1].to_array().squeeze() if hasattr(x_gen[-1], 'to_array') else x_gen[-1]
    )
    np.testing.assert_array_equal(gen_array, x_batch)  # type: ignore


@pytest.mark.parametrize(
    ('x_var', 'y_var'),
    [
        ('x', 'y'),  # xr.DataArray
        (['x'], ['y']),  # xr.Dataset
    ],
)
def test_map_dataset(ds_xy, x_var, y_var) -> None:
    x = ds_xy[x_var]
    y = ds_xy[y_var]

    x_gen = BatchGenerator(x, {'sample': 10})
    y_gen = BatchGenerator(y, {'sample': 10})

    dataset = MapDataset(x_gen, y_gen)

    # test __getitem__
    x_batch, y_batch = dataset[0]
    assert x_batch.shape == (10, 5)
    assert y_batch.shape == (10,)
    assert isinstance(x_batch, torch.Tensor)

    idx = torch.tensor([0])
    x_batch, y_batch = dataset[idx]
    assert x_batch.shape == (10, 5)
    assert y_batch.shape == (10,)
    assert isinstance(x_batch, torch.Tensor)

    with pytest.raises(NotImplementedError):
        idx = torch.tensor([0, 1])
        x_batch, y_batch = dataset[idx]

    # test __len__
    assert len(dataset) == len(x_gen)

    # test integration with torch DataLoader
    loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    for x_batch, y_batch in loader:
        assert x_batch.shape == (10, 5)
        assert y_batch.shape == (10,)
        assert isinstance(x_batch, torch.Tensor)

    # Check that array shape of last item in generator is same as the batch image
    assert tuple(x_gen[-1].sizes.values()) == x_batch.shape
    # Check that array values from last item in generator and batch are the same
    gen_array = (
        x_gen[-1].to_array().squeeze() if hasattr(x_gen[-1], 'to_array') else x_gen[-1]
    )
    np.testing.assert_array_equal(gen_array, x_batch)  # type: ignore


@pytest.mark.parametrize(
    ('x_var', 'y_var'),
    [
        ('x', 'y'),  # xr.DataArray
        (['x'], ['y']),  # xr.Dataset
    ],
)
def test_map_dataset_with_transform(ds_xy, x_var, y_var) -> None:
    x = ds_xy[x_var]
    y = ds_xy[y_var]

    x_gen = BatchGenerator(x, {'sample': 10})
    y_gen = BatchGenerator(y, {'sample': 10})

    def x_transform(batch):
        return to_tensor(batch * 0 + 1)

    def y_transform(batch):
        return to_tensor(batch * 0 - 1)

    dataset = MapDataset(
        x_gen, y_gen, transform=x_transform, target_transform=y_transform
    )
    x_batch, y_batch = dataset[0]
    assert x_batch.shape == (10, 5)
    assert y_batch.shape == (10,)
    assert isinstance(x_batch, torch.Tensor)
    assert (x_batch == 1).all()
    assert (y_batch == -1).all()


@pytest.mark.parametrize(
    ('x_var', 'y_var'),
    [
        ('x', 'y'),  # xr.DataArray
        (['x'], ['y']),  # xr.Dataset
    ],
)
def test_iterable_dataset(ds_xy, x_var, y_var):
    x = ds_xy[x_var]
    y = ds_xy[y_var]

    x_gen = BatchGenerator(x, {'sample': 10})
    y_gen = BatchGenerator(y, {'sample': 10})

    dataset = IterableDataset(x_gen, y_gen)

    # test integration with torch DataLoader
    loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    for x_batch, y_batch in loader:
        assert x_batch.shape == (10, 5)
        assert y_batch.shape == (10,)
        assert isinstance(x_batch, torch.Tensor)

    # Check that array shape of last item in generator is same as the batch image
    assert tuple(x_gen[-1].sizes.values()) == x_batch.shape
    # Check that array values from last item in generator and batch are the same
    gen_array = (
        x_gen[-1].to_array().squeeze() if hasattr(x_gen[-1], 'to_array') else x_gen[-1]
    )
    np.testing.assert_array_equal(gen_array, x_batch)
