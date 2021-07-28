import xarray as xr

from .generators import BatchGenerator


@xr.register_dataarray_accessor('batch')
@xr.register_dataset_accessor('batch')
class BatchAccessor:
    def __init__(self, xarray_obj):
        '''
        Batch accessor returning a BatchGenerator object via the `generator method`
        '''
        self._obj = xarray_obj

    def generator(self, *args, **kwargs):
        '''
        Return a BatchGenerator via the batch accessor

        Parameters
        ----------
        *args : iterable
            Positional arguments to pass to the `BatchGenerator` constructor.
        **kwargs : dict
            Keyword arguments to pass to the `BatchGenerator` constructor.
        '''
        return BatchGenerator(self._obj, *args, **kwargs)
