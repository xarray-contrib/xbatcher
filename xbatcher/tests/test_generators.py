import numpy as np
import pytest
import xarray as xr

from xbatcher import BatchGenerator


@pytest.fixture(scope="module")
def sample_ds_1d():
    """
    Sample 1D xarray.Dataset for testing.
    """
    size = 100
    ds = xr.Dataset(
        {
            "foo": (["x"], np.random.rand(size)),
            "bar": (["x"], np.random.randint(0, 10, size)),
        },
        {"x": (["x"], np.arange(size))},
    )
    return ds


@pytest.fixture(scope="module")
def sample_ds_3d():
    """
    Sample 3D xarray.Dataset for testing.
    """
    shape = (10, 50, 100)
    ds = xr.Dataset(
        {
            "foo": (["time", "y", "x"], np.random.rand(*shape)),
            "bar": (["time", "y", "x"], np.random.randint(0, 10, shape)),
        },
        {
            "x": (["x"], np.arange(shape[-1])),
            "y": (["y"], np.arange(shape[-2])),
        },
    )
    return ds


def test_constructor_dataarray():
    """
    Test that the xarray.DataArray passed to the batch generator is stored
    in the .ds attribute.
    """
    da = xr.DataArray(np.random.rand(10), dims="x", name="foo")
    bg = BatchGenerator(da, input_dims={"x": 2})
    xr.testing.assert_identical(da, bg.ds)


@pytest.mark.parametrize("input_size", [5, 6])
def test_batcher_length(sample_ds_1d, input_size):
    """ "
    Test the length of the batch generator.
    """
    bg = BatchGenerator(sample_ds_1d, input_dims={"x": input_size})
    assert len(bg) == sample_ds_1d.dims["x"] // input_size


def test_batcher_getitem(sample_ds_1d):
    """
    Test indexing on the batch generator.
    """
    bg = BatchGenerator(sample_ds_1d, input_dims={"x": 10})

    # first batch
    assert bg[0].dims["x"] == 10
    # last batch
    assert bg[-1].dims["x"] == 10
    # raises IndexError for out of range index
    with pytest.raises(IndexError, match=r"list index out of range"):
        bg[9999999]

    # raises NotImplementedError for iterable index
    with pytest.raises(NotImplementedError):
        bg[[1, 2, 3]]


@pytest.mark.parametrize("input_size", [5, 10])
def test_batch_1d(sample_ds_1d, input_size):
    """
    Test batch generation for a 1D dataset using ``input_dims``.
    """
    bg = BatchGenerator(sample_ds_1d, input_dims={"x": input_size})
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims["x"] == input_size
        expected_slice = slice(input_size * n, input_size * (n + 1))
        ds_batch_expected = sample_ds_1d.isel(x=expected_slice)
        xr.testing.assert_identical(ds_batch_expected, ds_batch)


@pytest.mark.parametrize("input_size", [5, 10])
def test_batch_1d_concat(sample_ds_1d, input_size):
    """
    Test batch generation for a 1D dataset using ``input_dims`` and concat_input_dims``.
    """
    bg = BatchGenerator(
        sample_ds_1d, input_dims={"x": input_size}, concat_input_dims=True
    )
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims["x_input"] == input_size
        assert ds_batch.dims["input_batch"] == sample_ds_1d.dims["x"] // input_size
        assert "x" in ds_batch.coords


@pytest.mark.parametrize("input_size", [5, 10])
def test_batch_1d_no_coordinate(sample_ds_1d, input_size):
    """
    Test batch generation for a 1D dataset without coordinates using ``input_dims``.

    Fix for https://github.com/xarray-contrib/xbatcher/issues/3.
    """
    ds_dropped = sample_ds_1d.drop_vars("x")
    bg = BatchGenerator(ds_dropped, input_dims={"x": input_size})
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims["x"] == input_size
        expected_slice = slice(input_size * n, input_size * (n + 1))
        ds_batch_expected = ds_dropped.isel(x=expected_slice)
        xr.testing.assert_identical(ds_batch_expected, ds_batch)


@pytest.mark.parametrize("input_size", [5, 10])
def test_batch_1d_concat_no_coordinate(sample_ds_1d, input_size):
    """
    Test batch generation for a 1D dataset without coordinates using ``input_dims``
    and ``concat_input_dims``.

    Fix for https://github.com/xarray-contrib/xbatcher/issues/3.
    """
    ds_dropped = sample_ds_1d.drop_vars("x")
    bg = BatchGenerator(
        ds_dropped, input_dims={"x": input_size}, concat_input_dims=True
    )
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims["x_input"] == input_size
        assert ds_batch.dims["input_batch"] == sample_ds_1d.dims["x"] // input_size
        assert "x" not in ds_batch.coords


@pytest.mark.parametrize("input_overlap", [1, 4])
def test_batch_1d_overlap(sample_ds_1d, input_overlap):
    """
    Test batch generation for a 1D dataset without coordinates using ``input_dims``
    and ``input_overlap``.
    """
    input_size = 10
    bg = BatchGenerator(
        sample_ds_1d, input_dims={"x": input_size}, input_overlap={"x": input_overlap}
    )
    stride = input_size - input_overlap
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims["x"] == input_size
        expected_slice = slice(stride * n, stride * n + input_size)
        ds_batch_expected = sample_ds_1d.isel(x=expected_slice)
        xr.testing.assert_identical(ds_batch_expected, ds_batch)


@pytest.mark.parametrize("input_size", [5, 10])
def test_batch_3d_1d_input(sample_ds_3d, input_size):
    """
    Test batch generation for a 3D dataset with 1 dimension
    specified in ``input_dims``.
    """
    bg = BatchGenerator(sample_ds_3d, input_dims={"x": input_size})
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims["x"] == input_size
        # time and y should be collapsed into batch dimension
        assert (
            ds_batch.dims["sample"]
            == sample_ds_3d.dims["y"] * sample_ds_3d.dims["time"]
        )
        expected_slice = slice(input_size * n, input_size * (n + 1))
        ds_batch_expected = (
            sample_ds_3d.isel(x=expected_slice)
            .stack(sample=["time", "y"])
            .transpose("sample", "x")
        )
        xr.testing.assert_identical(ds_batch_expected, ds_batch)


@pytest.mark.parametrize("input_size", [5, 10])
def test_batch_3d_2d_input(sample_ds_3d, input_size):
    """
    Test batch generation for a 3D dataset with 2 dimensions
    specified in ``input_dims``.
    """
    x_input_size = 20
    bg = BatchGenerator(sample_ds_3d, input_dims={"y": input_size, "x": x_input_size})
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims["x"] == x_input_size
        assert ds_batch.dims["y"] == input_size
        yn, xn = np.unravel_index(
            n,
            (
                (sample_ds_3d.dims["y"] // input_size),
                (sample_ds_3d.dims["x"] // x_input_size),
            ),
        )
        expected_xslice = slice(x_input_size * xn, x_input_size * (xn + 1))
        expected_yslice = slice(input_size * yn, input_size * (yn + 1))
        ds_batch_expected = sample_ds_3d.isel(x=expected_xslice, y=expected_yslice)
        xr.testing.assert_identical(ds_batch_expected, ds_batch)
    assert (n + 1) == (
        (sample_ds_3d.dims["x"] // x_input_size)
        * (sample_ds_3d.dims["y"] // input_size)
    )


@pytest.mark.parametrize("input_size", [5, 10])
def test_batch_3d_2d_input_concat(sample_ds_3d, input_size):
    """
    Test batch generation for a 3D dataset with 2 dimensions
    specified in ``input_dims`` using ``concat_input_dims``.
    """
    x_input_size = 20
    bg = BatchGenerator(
        sample_ds_3d,
        input_dims={"y": input_size, "x": x_input_size},
        concat_input_dims=True,
    )
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims["x_input"] == x_input_size
        assert ds_batch.dims["y_input"] == input_size
        assert ds_batch.dims["sample"] == (
            (sample_ds_3d.dims["x"] // x_input_size)
            * (sample_ds_3d.dims["y"] // input_size)
            * sample_ds_3d.dims["time"]
        )


def test_preload_batch_false(sample_ds_1d):
    """
    Test ``preload_batch=False`` does not compute Dask arrays.
    """
    sample_ds_1d_dask = sample_ds_1d.chunk({"x": 2})
    bg = BatchGenerator(sample_ds_1d_dask, input_dims={"x": 2}, preload_batch=False)
    assert bg.preload_batch is False
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.chunks


def test_preload_batch_true(sample_ds_1d):
    """
    Test ``preload_batch=True`` does computes Dask arrays.
    """
    sample_ds_1d_dask = sample_ds_1d.chunk({"x": 2})
    bg = BatchGenerator(sample_ds_1d_dask, input_dims={"x": 2}, preload_batch=True)
    assert bg.preload_batch is True
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        assert not ds_batch.chunks


def test_input_dim_exceptions(sample_ds_1d):
    """
    Test that a ValueError is raised when input_dim[dim] > ds.sizes[dim]
    """
    with pytest.raises(ValueError) as e:
        BatchGenerator(sample_ds_1d, input_dims={"x": 110})
        assert len(e) == 1


def test_input_overlap_exceptions(sample_ds_1d):
    """
    Test that a ValueError is raised when input_overlap[dim] > input_dim[dim]
    """
    with pytest.raises(ValueError) as e:
        BatchGenerator(sample_ds_1d, input_dims={"x": 10}, input_overlap={"x": 20})
        assert len(e) == 1
