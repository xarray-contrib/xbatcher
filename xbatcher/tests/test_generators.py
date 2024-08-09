import json
import tempfile
from typing import Any

import numpy as np
import pytest
import xarray as xr

from xbatcher import BatchGenerator, BatchSchema
from xbatcher.testing import (
    get_batch_dimensions,
    validate_batch_dimensions,
    validate_generator_length,
)


@pytest.fixture(scope='module')
def sample_ds_1d():
    """
    Sample 1D xarray.Dataset for testing.
    """
    size = 100
    ds = xr.Dataset(
        {
            'foo': (['x'], np.random.rand(size)),
            'bar': (['x'], np.random.randint(0, 10, size)),
        },
        {'x': (['x'], np.arange(size))},
    )
    return ds


@pytest.fixture(scope='module')
def sample_ds_3d():
    """
    Sample 3D xarray.Dataset for testing.
    """
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


def test_constructor_dataarray():
    """
    Test that the xarray.DataArray passed to the batch generator is stored
    in the .ds attribute.
    """
    da = xr.DataArray(np.random.rand(10), dims='x', name='foo')
    bg = BatchGenerator(da, input_dims={'x': 2})
    xr.testing.assert_identical(da, bg.ds)


@pytest.mark.parametrize('input_size', [5, 6])
def test_generator_length(sample_ds_1d, input_size):
    """ "
    Test the length of the batch generator.
    """
    bg = BatchGenerator(sample_ds_1d, input_dims={'x': input_size})
    validate_generator_length(bg)


def test_generator_getitem(sample_ds_1d):
    """
    Test indexing on the batch generator.
    """
    bg = BatchGenerator(sample_ds_1d, input_dims={'x': 10})
    first_batch = bg[0]
    last_batch = bg[-1]
    expected_dims = get_batch_dimensions(bg)
    validate_batch_dimensions(expected_dims=expected_dims, batch=first_batch)
    validate_batch_dimensions(expected_dims=expected_dims, batch=last_batch)
    # raises IndexError for out of range index
    with pytest.raises(IndexError, match=r'list index out of range'):
        bg[9999999]

    # raises NotImplementedError for iterable index
    with pytest.raises(NotImplementedError):
        bg[[1, 2, 3]]


@pytest.mark.parametrize('input_size', [5, 10])
def test_batch_1d(sample_ds_1d, input_size):
    """
    Test batch generation for a 1D dataset using ``input_dims``.
    """
    bg = BatchGenerator(sample_ds_1d, input_dims={'x': input_size})
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims['x'] == input_size
        expected_slice = slice(input_size * n, input_size * (n + 1))
        ds_batch_expected = sample_ds_1d.isel(x=expected_slice)
        xr.testing.assert_identical(ds_batch_expected, ds_batch)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


@pytest.mark.parametrize('input_size', [5, 10])
def test_batch_1d_concat(sample_ds_1d, input_size):
    """
    Test batch generation for a 1D dataset using ``input_dims`` and concat_input_dims``.
    """
    bg = BatchGenerator(
        sample_ds_1d, input_dims={'x': input_size}, concat_input_dims=True
    )
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)
        assert 'x' in ds_batch.coords


def test_batch_1d_concat_duplicate_dim(sample_ds_1d):
    """
    Test batch generation for a 1D dataset using ``concat_input_dims`` when
    the same dimension occurs in ``input_dims`` and `batch_dims``
    """
    bg = BatchGenerator(
        sample_ds_1d, input_dims={'x': 5}, batch_dims={'x': 10}, concat_input_dims=True
    )
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


@pytest.mark.parametrize('input_size', [5, 10])
def test_batch_1d_no_coordinate(sample_ds_1d, input_size):
    """
    Test batch generation for a 1D dataset without coordinates using ``input_dims``.

    Fix for https://github.com/xarray-contrib/xbatcher/issues/3.
    """
    ds_dropped = sample_ds_1d.drop_vars('x')
    bg = BatchGenerator(ds_dropped, input_dims={'x': input_size})
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims['x'] == input_size
        expected_slice = slice(input_size * n, input_size * (n + 1))
        ds_batch_expected = ds_dropped.isel(x=expected_slice)
        xr.testing.assert_identical(ds_batch_expected, ds_batch)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


@pytest.mark.parametrize('input_size', [5, 10])
def test_batch_1d_concat_no_coordinate(sample_ds_1d, input_size):
    """
    Test batch generation for a 1D dataset without coordinates using ``input_dims``
    and ``concat_input_dims``.

    Fix for https://github.com/xarray-contrib/xbatcher/issues/3.
    """
    ds_dropped = sample_ds_1d.drop_vars('x')
    bg = BatchGenerator(
        ds_dropped, input_dims={'x': input_size}, concat_input_dims=True
    )
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)
        assert 'x' not in ds_batch.coords


@pytest.mark.parametrize('input_overlap', [1, 4])
def test_batch_1d_overlap(sample_ds_1d, input_overlap):
    """
    Test batch generation for a 1D dataset without coordinates using ``input_dims``
    and ``input_overlap``.
    """
    input_size = 10
    bg = BatchGenerator(
        sample_ds_1d, input_dims={'x': input_size}, input_overlap={'x': input_overlap}
    )
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    stride = input_size - input_overlap
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims['x'] == input_size
        expected_slice = slice(stride * n, stride * n + input_size)
        ds_batch_expected = sample_ds_1d.isel(x=expected_slice)
        xr.testing.assert_identical(ds_batch_expected, ds_batch)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


@pytest.mark.parametrize('input_size', [5, 10])
def test_batch_3d_1d_input(sample_ds_3d, input_size):
    """
    Test batch generation for a 3D dataset with 1 dimension
    specified in ``input_dims``.
    """
    bg = BatchGenerator(sample_ds_3d, input_dims={'x': input_size})
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims['x'] == input_size
        # time and y should be collapsed into batch dimension
        assert (
            ds_batch.dims['sample']
            == sample_ds_3d.dims['y'] * sample_ds_3d.dims['time']
        )
        expected_slice = slice(input_size * n, input_size * (n + 1))
        ds_batch_expected = (
            sample_ds_3d.isel(x=expected_slice)
            .stack(sample=['time', 'y'])
            .transpose('sample', 'x')
        )
        xr.testing.assert_identical(ds_batch_expected, ds_batch)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


@pytest.mark.parametrize(
    'concat',
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason='Bug described in https://github.com/xarray-contrib/xbatcher/issues/126'
            ),
        ),
    ],
)
def test_batch_3d_1d_input_batch_dims(sample_ds_3d, concat):
    """
    Test batch generation for a 3D dataset using ``input_dims`` and batch_dims``.
    """
    bg = BatchGenerator(
        sample_ds_3d,
        input_dims={'x': 5, 'y': 10},
        batch_dims={'time': 2},
        concat_input_dims=concat,
    )
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for ds_batch in bg:
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


def test_batch_3d_1d_input_batch_concat_duplicate_dim(sample_ds_3d):
    """
    Test batch generation for a 3D dataset using ``concat_input_dims`` when
    the same dimension occurs in ``input_dims`` and batch_dims``.
    """
    bg = BatchGenerator(
        sample_ds_3d,
        input_dims={'x': 5, 'y': 10},
        batch_dims={'x': 10, 'y': 20},
        concat_input_dims=True,
    )
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for ds_batch in bg:
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


@pytest.mark.parametrize('input_size', [5, 10])
def test_batch_3d_2d_input(sample_ds_3d, input_size):
    """
    Test batch generation for a 3D dataset with 2 dimensions
    specified in ``input_dims``.
    """
    x_input_size = 20
    bg = BatchGenerator(sample_ds_3d, input_dims={'y': input_size, 'x': x_input_size})
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for n, ds_batch in enumerate(bg):
        yn, xn = np.unravel_index(
            n,
            (
                (sample_ds_3d.dims['y'] // input_size),
                (sample_ds_3d.dims['x'] // x_input_size),
            ),
        )
        expected_xslice = slice(x_input_size * xn, x_input_size * (xn + 1))
        expected_yslice = slice(input_size * yn, input_size * (yn + 1))
        ds_batch_expected = sample_ds_3d.isel(x=expected_xslice, y=expected_yslice)
        xr.testing.assert_identical(ds_batch_expected, ds_batch)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


@pytest.mark.parametrize('input_size', [5, 10])
def test_batch_3d_2d_input_concat(sample_ds_3d, input_size):
    """
    Test batch generation for a 3D dataset with 2 dimensions
    specified in ``input_dims`` using ``concat_input_dims``.
    """
    x_input_size = 20
    bg = BatchGenerator(
        sample_ds_3d,
        input_dims={'y': input_size, 'x': x_input_size},
        concat_input_dims=True,
    )
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)

    bg = BatchGenerator(
        sample_ds_3d,
        input_dims={'time': input_size, 'x': x_input_size},
        concat_input_dims=True,
    )
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


def test_preload_batch_false(sample_ds_1d):
    """
    Test ``preload_batch=False`` does not compute Dask arrays.
    """
    sample_ds_1d_dask = sample_ds_1d.chunk({'x': 2})
    bg = BatchGenerator(sample_ds_1d_dask, input_dims={'x': 2}, preload_batch=False)
    assert bg.preload_batch is False
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.chunks


def test_preload_batch_true(sample_ds_1d):
    """
    Test ``preload_batch=True`` does computes Dask arrays.
    """
    sample_ds_1d_dask = sample_ds_1d.chunk({'x': 2})
    bg = BatchGenerator(sample_ds_1d_dask, input_dims={'x': 2}, preload_batch=True)
    assert bg.preload_batch is True
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        assert not ds_batch.chunks


def test_input_dim_exceptions(sample_ds_1d):
    """
    Test that a ValueError is raised when input_dim[dim] > ds.sizes[dim]
    """
    with pytest.raises(ValueError) as e:
        BatchGenerator(sample_ds_1d, input_dims={'x': 110})
        assert len(e) == 1


def test_input_overlap_exceptions(sample_ds_1d):
    """
    Test that a ValueError is raised when input_overlap[dim] > input_dim[dim]
    """
    with pytest.raises(ValueError) as e:
        BatchGenerator(sample_ds_1d, input_dims={'x': 10}, input_overlap={'x': 20})
        assert len(e) == 1


@pytest.mark.parametrize('input_size', [5, 10])
def test_to_json(sample_ds_3d, input_size):
    x_input_size = 20
    bg = BatchSchema(
        sample_ds_3d,
        input_dims={'time': input_size, 'x': x_input_size},
    )
    out_file = tempfile.NamedTemporaryFile(mode='w+b')
    bg.to_file(out_file.name)
    in_dict = json.load(out_file)
    assert in_dict['input_dims']['time'] == input_size
    assert in_dict['input_dims']['x'] == x_input_size
    out_file.close()


@pytest.mark.parametrize('preload', [True, False])
def test_batcher_cached_getitem(sample_ds_1d, preload) -> None:
    pytest.importorskip('zarr')
    cache: dict[str, Any] = {}

    def preproc(ds):
        processed = ds.load().chunk(-1)
        processed.attrs['foo'] = 'bar'
        return processed

    bg = BatchGenerator(
        sample_ds_1d,
        input_dims={'x': 10},
        cache=cache,
        cache_preprocess=preproc,
        preload_batch=preload,
    )

    # first batch
    assert bg[0].sizes['x'] == 10
    ds_no_cache = bg[1]
    # last batch
    assert bg[-1].sizes['x'] == 10

    assert '0/.zgroup' in cache

    # now from cache
    # first batch
    assert bg[0].sizes['x'] == 10
    # last batch
    assert bg[-1].sizes['x'] == 10
    ds_cache = bg[1]

    assert ds_no_cache.attrs['foo'] == 'bar'
    assert ds_cache.attrs['foo'] == 'bar'

    xr.testing.assert_equal(ds_no_cache, ds_cache)
    xr.testing.assert_identical(ds_no_cache, ds_cache)

    # without preprocess func
    bg = BatchGenerator(
        sample_ds_1d, input_dims={'x': 10}, cache=cache, preload_batch=preload
    )
    assert bg.cache_preprocess is None
    assert bg[0].sizes['x'] == 10
    ds_no_cache = bg[1]
    assert '1/.zgroup' in cache
    ds_cache = bg[1]
    xr.testing.assert_equal(ds_no_cache, ds_cache)
    xr.testing.assert_identical(ds_no_cache, ds_cache)
