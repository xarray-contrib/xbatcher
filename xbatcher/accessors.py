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


@xr.register_dataarray_accessor('torch')
class TorchAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_tensor(self):
        """Convert this DataArray to a torch.Tensor"""
        import torch

        return torch.tensor(self._obj.data)

    def to_named_tensor(self):
        """Convert this DataArray to a torch.Tensor with named dimensions"""
        import torch

        return torch.tensor(self._obj.data, names=self._obj.sizes)
