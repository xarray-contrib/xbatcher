import xarray as xr
import numpy as np
import tempfile
import xbatcher


def test_to_zarr():
    da = xr.DataArray(
        np.random.rand(1000, 100, 100), name="foo", dims=["time", "y", "x"]
    ).chunk({"time": 1})

    bgen = xbatcher.BatchGenerator(da, {"time": 10}, preload_batch=False)

    for ds_batch in bgen:
        ds_first_batch = ds_batch
        break

    tempdir = tempfile.TemporaryDirectory().name
    bgen.to_zarr(tempdir)

    bgen_loaded = xbatcher.BatchGenerator.from_zarr(tempdir)

    for loaded_batch in bgen_loaded:
        loaded_first_batch = loaded_batch
        break

    # DataArray.equals doesn't work while the DataArray's are still stacked
    da_first_batch = ds_first_batch.unstack()
    da_loaded_first_batch = loaded_first_batch.unstack()
    # For some reason DataArray.equals doesn't work here, but DataArray.broadcast_equals did
    assert da_loaded_first_batch.broadcast_equals(da_first_batch)
    # I think this should mean that DataArray.equals should work
    assert (da_loaded_first_batch - da_first_batch).max() == 0.0
