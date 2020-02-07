import xarray as xr

from .generators import BatchGenerator


@xr.register_dataarray_accessor("batch")
@xr.register_dataset_accessor("batch")
class BatchAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def generator(self, *args, **kwargs):
        return BatchGenerator(self._obj, *args, **kwargs)
