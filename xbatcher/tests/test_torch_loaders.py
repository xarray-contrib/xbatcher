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
    assert len(x_batch) == len(y_batch)
    assert isinstance(x_batch, np.ndarray)

    # test __len__
    assert len(dataset) == len(x_gen)

    # test integration with torch DataLoader
    loader = torch.utils.data.DataLoader(dataset)

    for x_batch, y_batch in loader:
        assert len(x_batch) == len(y_batch)
        assert isinstance(x_batch, torch.Tensor)

    # TODO: why does pytorch add an extra dimension (length 1) to x_batch
    assert x_gen[-1]['x'].shape == x_batch.shape[1:]
    # TODO: also need to revisit the variable extraction bits here
    assert np.array_equal(x_gen[-1]['x'], x_batch[0, :, :])


def test_iterable_dataset(ds_xy):

    x = ds_xy['x']
    y = ds_xy['y']

    x_gen = BatchGenerator(x, {'sample': 10})
    y_gen = BatchGenerator(y, {'sample': 10})

    dataset = IterableDataset(x_gen, y_gen)

    # test integration with torch DataLoader
    loader = torch.utils.data.DataLoader(dataset)

    for x_batch, y_batch in loader:
        assert len(x_batch) == len(y_batch)
        assert isinstance(x_batch, torch.Tensor)

    # TODO: why does pytorch add an extra dimension (length 1) to x_batch
    assert x_gen[-1]['x'].shape == x_batch.shape[1:]
    # TODO: also need to revisit the variable extraction bits here
    assert np.array_equal(x_gen[-1]['x'], x_batch[0, :, :])
