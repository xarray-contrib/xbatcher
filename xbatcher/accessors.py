import xarray as xr

from .generators import BatchGenerator


@xr.register_dataarray_accessor("batch")
@xr.register_dataset_accessor("batch")
class BatchAccessor:
    def __init__(self, xarray_obj):
        """
        Batch accessor returning a BatchGenerator object via the `generator method`
        """
        self._obj = xarray_obj

    def generator(self, *args, **kwargs):
        """
        Return a BatchGenerator via the batch accessor

        Parameters
        ----------
        *args : iterable
            Positional arguments to pass to the `BatchGenerator` constructor.
        **kwargs : dict
            Keyword arguments to pass to the `BatchGenerator` constructor.
        """
        return BatchGenerator(self._obj, *args, **kwargs)


@xr.register_dataarray_accessor("torch")
@xr.register_dataset_accessor("torch")
class TorchAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _as_xarray_dataarray(self, xr_obj):
        """
        Convert xarray.Dataset to xarray.DataArray if needed, so that it can
        be converted into a torch.Tensor object.
        """
        try:
            # Convert xr.Dataset to xr.DataArray
            dataarray = xr_obj.to_array().squeeze(dim="variable")
        except AttributeError:  # 'DataArray' object has no attribute 'to_array'
            # If object is already an xr.DataArray
            dataarray = xr_obj

        return dataarray

    def to_tensor(self):
        """Convert this DataArray to a torch.Tensor"""
        import torch

        dataarray = self._as_xarray_dataarray(xr_obj=self._obj)

        return torch.tensor(data=dataarray.data)

    def to_named_tensor(self):
        """
        Convert this DataArray to a torch.Tensor with named dimensions.

        See https://pytorch.org/docs/stable/named_tensor.html
        """
        import torch

        dataarray = self._as_xarray_dataarray(xr_obj=self._obj)

        return torch.tensor(data=dataarray.data, names=tuple(dataarray.sizes))
