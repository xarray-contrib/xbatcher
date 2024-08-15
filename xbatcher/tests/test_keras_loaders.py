import numpy as np
import pytest
import xarray as xr

from xbatcher import BatchGenerator
from xbatcher.loaders.keras import CustomTFDataset

tf = pytest.importorskip('tensorflow')


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


def test_custom_dataarray(ds_xy):
    x = ds_xy['x']
    y = ds_xy['y']

    x_gen = BatchGenerator(x, {'sample': 10})
    y_gen = BatchGenerator(y, {'sample': 10})

    dataset = CustomTFDataset(x_gen, y_gen)

    # test __getitem__
    x_batch, y_batch = dataset[0]
    assert x_batch.shape == (10, 5)
    assert y_batch.shape == (10,)
    assert tf.is_tensor(x_batch)
    assert tf.is_tensor(y_batch)

    # test __len__
    assert len(dataset) == len(x_gen)


def test_custom_dataarray_with_transform(ds_xy):
    x = ds_xy['x']
    y = ds_xy['y']

    x_gen = BatchGenerator(x, {'sample': 10})
    y_gen = BatchGenerator(y, {'sample': 10})

    def x_transform(batch):
        return batch * 0 + 1

    def y_transform(batch):
        return batch * 0 - 1

    dataset = CustomTFDataset(
        x_gen, y_gen, transform=x_transform, target_transform=y_transform
    )
    x_batch, y_batch = dataset[0]
    assert x_batch.shape == (10, 5)
    assert y_batch.shape == (10,)
    assert tf.is_tensor(x_batch)
    assert tf.is_tensor(y_batch)
    assert tf.experimental.numpy.all(x_batch == 1)
    assert tf.experimental.numpy.all(y_batch == -1)
