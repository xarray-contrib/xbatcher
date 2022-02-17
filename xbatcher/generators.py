"""Classes for iterating through xarray datarrays / datasets in batches."""

import itertools
from collections import OrderedDict

import xarray as xr


def _as_xarray_dataset(ds):
    # maybe coerce to xarray dataset
    if isinstance(ds, xr.Dataset):
        return ds
    else:
        return ds.to_dataset()


def _slices(dimsize, size, overlap=0):
    # return a list of slices to chop up a single dimension
    slices = []
    stride = size - overlap
    assert stride > 0
    assert stride <= dimsize
    for start in range(0, dimsize, stride):
        end = start + size
        if end <= dimsize:
            slices.append(slice(start, end))
    return slices


def _iterate_through_dataset(ds, dims, overlap={}):
    dim_slices = []
    for dim in dims:
        dimsize = ds.dims[dim]
        size = dims[dim]
        olap = overlap.get(dim, 0)
        dim_slices.append(_slices(dimsize, size, olap))

    for slices in itertools.product(*dim_slices):
        selector = {key: slice for key, slice in zip(dims, slices)}
        yield ds.isel(**selector)


def _drop_input_dims(ds, input_dims, suffix='_input'):
    # remove input_dims coordinates from datasets, rename the dimensions
    # then put intput_dims back in as coordinates
    out = ds.copy()
    for dim in input_dims:
        newdim = dim + suffix
        out = out.rename({dim: newdim})
        # extra steps needed if there is a coordinate
        if newdim in out:
            out = out.drop_vars(newdim)
            out.coords[dim] = newdim, ds[dim].data, ds[dim].attrs
    return out


def _maybe_stack_batch_dims(
    ds, input_dims, squeeze_batch_dim, stacked_dim_name='sample'
):
    batch_dims = [d for d in ds.dims if d not in input_dims]
    if len(batch_dims) == 0:
        if squeeze_batch_dim:
            return ds
        else:
            return ds.expand_dims(stacked_dim_name, 0)
    elif len(batch_dims) == 1:
        return ds
    else:
        ds_stack = ds.stack(**{stacked_dim_name: batch_dims})
        # ensure correct order
        dim_order = (stacked_dim_name,) + tuple(input_dims)
        return ds_stack.transpose(*dim_order)


class BatchGenerator:
    """Create generator for iterating through xarray datarrays / datasets in
    batches.

    Parameters
    ----------
    ds : ``xarray.Dataset`` or ``xarray.DataArray``
        The data to iterate over
    input_dims : dict
        A dictionary specifying the size of the inputs in each dimension,
        e.g. ``{'lat': 30, 'lon': 30}``
        These are the dimensions the ML library will see. All other dimensions
        will be stacked into one dimension called ``batch``.
    input_overlap : dict, optional
        A dictionary specifying the overlap along each dimension
        e.g. ``{'lat': 3, 'lon': 3}``
    batch_dims : dict, optional
        A dictionary specifying the size of the batch along each dimension
        e.g. ``{'time': 10}``. These will always be interated over.
    concat_input_dims : bool, optional
        If ``True``, the dimension chunks specified in ``input_dims`` will be
        concatenated and stacked into the batch dimension. If ``False``, they
        will be iterated over.
    preload_batch : bool, optional
        If ``True``, each batch will be loaded into memory before reshaping /
        processing, triggering any dask arrays to be computed.
    squeeze_batch_dim : bool, optional
        If ``False`` and all dims are input dims, each batch's dataset will have a
        "batch" dimension of size 1 prepended to the array. This functionality is
        useful for interoperability with Keras / Tensorflow.

    Yields
    ------
    ds_slice : ``xarray.Dataset`` or ``xarray.DataArray``
        Slices of the array matching the given batch size specification.
    """

    def __init__(
        self,
        ds,
        input_dims,
        input_overlap={},
        batch_dims={},
        concat_input_dims=False,
        preload_batch=True,
        squeeze_batch_dim=True,
    ):

        self.ds = _as_xarray_dataset(ds)
        # should be a dict
        self.input_dims = OrderedDict(input_dims)
        self.input_overlap = input_overlap
        self.batch_dims = OrderedDict(batch_dims)
        self.concat_input_dims = concat_input_dims
        self.preload_batch = preload_batch
        self.squeeze_batch_dim = squeeze_batch_dim

    def __iter__(self):
        for ds_batch in self._iterate_batch_dims(self.ds):
            if self.preload_batch:
                ds_batch.load()
            input_generator = self._iterate_input_dims(ds_batch)
            if self.concat_input_dims:
                new_dim_suffix = '_input'
                all_dsets = [
                    _drop_input_dims(
                        ds_input, list(self.input_dims), suffix=new_dim_suffix
                    )
                    for ds_input in input_generator
                ]
                dsc = xr.concat(all_dsets, dim='input_batch')
                new_input_dims = [
                    dim + new_dim_suffix for dim in self.input_dims
                ]
                yield _maybe_stack_batch_dims(
                    dsc, new_input_dims, self.squeeze_batch_dim
                )
            else:
                for ds_input in input_generator:
                    yield _maybe_stack_batch_dims(
                        ds_input, list(self.input_dims), self.squeeze_batch_dim
                    )

    def _iterate_batch_dims(self, ds):
        return _iterate_through_dataset(ds, self.batch_dims)

    def _iterate_input_dims(self, ds):
        return _iterate_through_dataset(ds, self.input_dims, self.input_overlap)
