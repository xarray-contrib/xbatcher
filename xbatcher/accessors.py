from typing import Any

import xarray as xr

from .generators import BatchGenerator


def _as_xarray_dataarray(xr_obj: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """
    Convert xarray.Dataset to xarray.DataArray if needed, so that it can
    be converted into a Tensor object.
    """
    if isinstance(xr_obj, xr.Dataset):
        xr_obj = xr_obj.to_array().squeeze(dim='variable')

    return xr_obj


@xr.register_dataarray_accessor('batch')
@xr.register_dataset_accessor('batch')
class BatchAccessor:
    def __init__(self, xarray_obj: xr.Dataset | xr.DataArray):
        """
        Batch accessor returning a BatchGenerator object via the `generator method`
        """
        self._obj = xarray_obj

    def generator(self, *args, **kwargs) -> BatchGenerator:
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


@xr.register_dataarray_accessor('tf')
@xr.register_dataset_accessor('tf')
class TFAccessor:
    def __init__(self, xarray_obj: xr.Dataset | xr.DataArray):
        self._obj = xarray_obj

    def to_tensor(self) -> Any:
        """Convert this DataArray to a tensorflow.Tensor"""
        import tensorflow as tf

        dataarray = _as_xarray_dataarray(xr_obj=self._obj)

        return tf.convert_to_tensor(dataarray.data)


@xr.register_dataarray_accessor('torch')
@xr.register_dataset_accessor('torch')
class TorchAccessor:
    def __init__(self, xarray_obj: xr.Dataset | xr.DataArray):
        self._obj = xarray_obj

    def to_tensor(self) -> Any:
        """Convert this DataArray to a torch.Tensor"""
        import torch

        dataarray = _as_xarray_dataarray(xr_obj=self._obj)

        return torch.tensor(data=dataarray.data)

    def to_named_tensor(self) -> Any:
        """
        Convert this DataArray to a torch.Tensor with named dimensions.

        See https://pytorch.org/docs/stable/named_tensor.html
        """
        import torch

        dataarray = _as_xarray_dataarray(xr_obj=self._obj)

        return torch.tensor(data=dataarray.data, names=tuple(dataarray.sizes))
