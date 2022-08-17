import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip('torch')

from xbatcher import BatchGenerator
from xbatcher.loaders.torch import IterableDataset, MapDataset


@pytest.fixture(scope='module')
def ds_xy():
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
    return ds


def test_map_dataset(ds_xy):

    x = ds_xy['x']
    y = ds_xy['y']

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
    loader = torch.utils.data.DataLoader(dataset)

    for x_batch, y_batch in loader:
        assert x_batch.shape == (1, 10, 5)
        assert y_batch.shape == (1, 10)
        assert isinstance(x_batch, torch.Tensor)

    # TODO: why does pytorch add an extra dimension (length 1) to x_batch
    assert x_gen[-1].shape == x_batch.shape[1:]
    # TODO: add test for xarray.Dataset
    assert np.array_equal(x_gen[-1], x_batch[0, :, :])


def test_map_dataset_with_transform(ds_xy):

    x = ds_xy['x']
    y = ds_xy['y']

    x_gen = BatchGenerator(x, {'sample': 10})
    y_gen = BatchGenerator(y, {'sample': 10})

    def x_transform(batch):
        return batch * 0 + 1

    def y_transform(batch):
        return batch * 0 - 1

    dataset = MapDataset(
        x_gen, y_gen, transform=x_transform, target_transform=y_transform
    )
    x_batch, y_batch = dataset[0]
    assert x_batch.shape == (10, 5)
    assert y_batch.shape == (10,)
    assert isinstance(x_batch, torch.Tensor)
    assert (x_batch == 1).all()
    assert (y_batch == -1).all()


def test_iterable_dataset(ds_xy):

    x = ds_xy['x']
    y = ds_xy['y']

    x_gen = BatchGenerator(x, {'sample': 10})
    y_gen = BatchGenerator(y, {'sample': 10})

    dataset = IterableDataset(x_gen, y_gen)

    # test integration with torch DataLoader
    loader = torch.utils.data.DataLoader(dataset)

    for x_batch, y_batch in loader:
        assert x_batch.shape == (1, 10, 5)
        assert y_batch.shape == (1, 10)
        assert isinstance(x_batch, torch.Tensor)

    # TODO: why does pytorch add an extra dimension (length 1) to x_batch
    assert x_gen[-1].shape == x_batch.shape[1:]
    # TODO: add test for xarray.Dataset
    assert np.array_equal(x_gen[-1], x_batch[0, :, :])
