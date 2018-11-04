"""Classes for iterating through xarray datarrays / datasets in batches."""

import xarray as xr
from collections import OrderedDict
import itertools

def _as_xarray_dataset(ds):
    # maybe coerce to xarray dataset
    if isinstance(ds, xr.Dataset):
        return ds
    else:
        return ds.to_dataset()


class BatchGenerator:
    """Create generator for iterating through xarray datarrays / datasets in
    batches.

    Parameters
    ----------
    ds : ``xarray.Dataset`` or ``xarray.DataArray``
        The data to iterate over
    batch_sizes : dict
        A dictionary specifying the size of the batch in each dimension,
        e.g. ``{'time': 100, 'latitude': 30}``
    overlap : dict, optional
        A dictionary specifying the overlap along each dimension

    Yields
    ------
    ds_slice : ``xarray.Dataset`` or ``xarray.DataArray``
        Slices of the array matching the given batch size specification
    """

    def __init__(self, ds, batch_sizes, overlap={}):
        self.ds = _as_xarray_dataset(ds)
        # should be a dict
        self.batch_sizes = OrderedDict(batch_sizes)
        self.batch_dims = list(self.batch_sizes)
        # make overlap is defined for each batch size defined
        self.overlap = {k: overlap.get(k, 0) for k in self.batch_dims}


    def __iter__(self):
        for slices in itertools.product(*[self._iterate_dim(dim)
                                          for dim in self.batch_dims]):
            selector = {key: slice for key, slice in zip(self.batch_dims, slices)}
            yield self.ds.isel(**selector)


    def _iterate_dim(self, dim):
        dimsize = self.ds.dims[dim]
        size = self.batch_sizes[dim]
        overlap = self.overlap[dim]
        stride = size - overlap
        assert stride > 0
        assert stride < dimsize
        for start in range(0, dimsize, stride):
            end = start+size
            if end <= dimsize:
                yield slice(start, end)
            else:
                return
