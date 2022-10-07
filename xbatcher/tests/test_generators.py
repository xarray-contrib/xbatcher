import numpy as np
import pytest
import xarray as xr

from xbatcher import BatchGenerator


@pytest.fixture(scope="module")
def sample_ds_1d():
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
    da = xr.DataArray(np.random.rand(10), dims="x", name="foo")
    bg = BatchGenerator(da, input_dims={"x": 2})
    assert isinstance(bg.ds, xr.DataArray)
    assert bg.ds.equals(da)


@pytest.mark.parametrize("bsize", [5, 6])
def test_batcher_lenth(sample_ds_1d, bsize):
    bg = BatchGenerator(sample_ds_1d, input_dims={"x": bsize})
    assert len(bg) == sample_ds_1d.dims["x"] // bsize


def test_batcher_getitem(sample_ds_1d):
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


# TODO: decide how to handle bsizes like 15 that don't evenly divide the dimension
# Should we enforce that each batch size always has to be the same
@pytest.mark.parametrize("bsize", [5, 10])
def test_batch_1d(sample_ds_1d, bsize):
    bg = BatchGenerator(sample_ds_1d, input_dims={"x": bsize})
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        # TODO: maybe relax this? see comment above
        assert ds_batch.dims["x"] == bsize
        expected_slice = slice(bsize * n, bsize * (n + 1))
        ds_batch_expected = sample_ds_1d.isel(x=expected_slice)
        assert ds_batch.equals(ds_batch_expected)


@pytest.mark.parametrize("bsize", [5, 10])
def test_batch_1d_concat(sample_ds_1d, bsize):
    bg = BatchGenerator(sample_ds_1d, input_dims={"x": bsize}, concat_input_dims=True)
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims["x_input"] == bsize
        assert ds_batch.dims["input_batch"] == sample_ds_1d.dims["x"] // bsize
        assert "x" in ds_batch.coords


@pytest.mark.parametrize("bsize", [5, 10])
def test_batch_1d_no_coordinate(sample_ds_1d, bsize):
    # fix for #3
    ds_dropped = sample_ds_1d.drop_vars("x")
    bg = BatchGenerator(ds_dropped, input_dims={"x": bsize})
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims["x"] == bsize
        expected_slice = slice(bsize * n, bsize * (n + 1))
        ds_batch_expected = ds_dropped.isel(x=expected_slice)
        assert ds_batch.equals(ds_batch_expected)


@pytest.mark.parametrize("bsize", [5, 10])
def test_batch_1d_concat_no_coordinate(sample_ds_1d, bsize):
    # test for #3
    ds_dropped = sample_ds_1d.drop_vars("x")
    bg = BatchGenerator(ds_dropped, input_dims={"x": bsize}, concat_input_dims=True)
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims["x_input"] == bsize
        assert ds_batch.dims["input_batch"] == sample_ds_1d.dims["x"] // bsize
        assert "x" not in ds_batch.coords


@pytest.mark.parametrize("olap", [1, 4])
def test_batch_1d_overlap(sample_ds_1d, olap):
    bsize = 10
    bg = BatchGenerator(
        sample_ds_1d, input_dims={"x": bsize}, input_overlap={"x": olap}
    )
    stride = bsize - olap
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims["x"] == bsize
        expected_slice = slice(stride * n, stride * n + bsize)
        ds_batch_expected = sample_ds_1d.isel(x=expected_slice)
        assert ds_batch.equals(ds_batch_expected)


@pytest.mark.parametrize("bsize", [5, 10])
def test_batch_3d_1d_input(sample_ds_3d, bsize):
    # first do the iteration over just one dimension
    bg = BatchGenerator(sample_ds_3d, input_dims={"x": bsize})
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims["x"] == bsize
        # time and y should be collapsed into batch dimension
        assert (
            ds_batch.dims["sample"]
            == sample_ds_3d.dims["y"] * sample_ds_3d.dims["time"]
        )
        expected_slice = slice(bsize * n, bsize * (n + 1))
        ds_batch_expected = (
            sample_ds_3d.isel(x=expected_slice)
            .stack(sample=["time", "y"])
            .transpose("sample", "x")
        )
        print(ds_batch)
        print(ds_batch_expected)
        assert ds_batch.equals(ds_batch_expected)


@pytest.mark.parametrize("bsize", [5, 10])
def test_batch_3d_2d_input(sample_ds_3d, bsize):
    # now iterate over both
    xbsize = 20
    bg = BatchGenerator(sample_ds_3d, input_dims={"y": bsize, "x": xbsize})
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims["x"] == xbsize
        assert ds_batch.dims["y"] == bsize
        yn, xn = np.unravel_index(
            n,
            (
                (sample_ds_3d.dims["y"] // bsize),
                (sample_ds_3d.dims["x"] // xbsize),
            ),
        )
        expected_xslice = slice(xbsize * xn, xbsize * (xn + 1))
        expected_yslice = slice(bsize * yn, bsize * (yn + 1))
        ds_batch_expected = sample_ds_3d.isel(x=expected_xslice, y=expected_yslice)
        xr.testing.assert_equal(ds_batch_expected, ds_batch)
    assert (n + 1) == (
        (sample_ds_3d.dims["x"] // xbsize) * (sample_ds_3d.dims["y"] // bsize)
    )


@pytest.mark.parametrize("bsize", [5, 10])
def test_batch_3d_2d_input_concat(sample_ds_3d, bsize):
    # now iterate over both
    xbsize = 20
    bg = BatchGenerator(
        sample_ds_3d,
        input_dims={"y": bsize, "x": xbsize},
        concat_input_dims=True,
    )
    for n, ds_batch in enumerate(bg):
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.dims["x_input"] == xbsize
        assert ds_batch.dims["y_input"] == bsize
        assert ds_batch.dims["sample"] == (
            (sample_ds_3d.dims["x"] // xbsize)
            * (sample_ds_3d.dims["y"] // bsize)
            * sample_ds_3d.dims["time"]
        )


def test_preload_batch_false(sample_ds_1d):
    sample_ds_1d_dask = sample_ds_1d.chunk({"x": 2})
    bg = BatchGenerator(sample_ds_1d_dask, input_dims={"x": 2}, preload_batch=False)
    assert bg.preload_batch is False
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.chunks


def test_preload_batch_true(sample_ds_1d):
    sample_ds_1d_dask = sample_ds_1d.chunk({"x": 2})
    bg = BatchGenerator(sample_ds_1d_dask, input_dims={"x": 2}, preload_batch=True)
    assert bg.preload_batch is True
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        assert not ds_batch.chunks


def test_batch_exceptions(sample_ds_1d):
    # ValueError when input_dim[dim] > ds.sizes[dim]
    with pytest.raises(ValueError) as e:
        BatchGenerator(sample_ds_1d, input_dims={"x": 110})
        assert len(e) == 1
    # ValueError when input_overlap[dim] > input_dim[dim]
    with pytest.raises(ValueError) as e:
        BatchGenerator(sample_ds_1d, input_dims={"x": 10}, input_overlap={"x": 20})
        assert len(e) == 1
