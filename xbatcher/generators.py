"""Classes for iterating through xarray datarrays / datasets in batches."""

import itertools
import json
from collections import OrderedDict
from typing import Any, Dict, Hashable, Iterator

import xarray as xr


def _as_xarray_dataset(ds):
    # maybe coerce to xarray dataset
    if isinstance(ds, xr.Dataset):
        return ds
    else:
        return ds.to_dataset()


def _slices(dimsize, size, overlap=0):
    # return a list of slices to chop up a single dimension
    if overlap >= size:
        raise ValueError(
            'input overlap must be less than the input sample length, but '
            f'the input sample length is {size} and the overlap is {overlap}'
        )
    slices = []
    stride = size - overlap
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
        if size > dimsize:
            raise ValueError(
                'input sample length must be less than or equal to the '
                f'dimension length, but the sample length of {size} '
                f'is greater than the dimension length of {dimsize} '
                f'for {dim}'
            )
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


def _maybe_stack_batch_dims(ds, input_dims, stacked_dim_name='sample'):
    batch_dims = [d for d in ds.dims if d not in input_dims]
    if len(batch_dims) < 2:
        return ds
    ds_stack = ds.stack(**{stacked_dim_name: batch_dims})
    # ensure correct order
    dim_order = (stacked_dim_name,) + tuple(input_dims)
    return ds_stack.transpose(*dim_order)


class BatchGeneratorBase:
    def __init__(
        self,
        input_dims: Dict[Hashable, int],
        input_overlap: Dict[Hashable, int] = {},
        batch_dims: Dict[Hashable, int] = {},
        concat_input_dims: bool = False,
    ):
        self.input_dims = OrderedDict(input_dims)
        self.input_overlap = input_overlap
        self.batch_dims = OrderedDict(batch_dims)
        self.concat_input_dims = concat_input_dims


class BatchGenerator(BatchGeneratorBase):
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
        will be stacked into one dimension called ``sample``.
    input_overlap : dict, optional
        A dictionary specifying the overlap along each dimension
        e.g. ``{'lat': 3, 'lon': 3}``
    batch_dims : dict, optional
        A dictionary specifying the size of the batch along each dimension
        e.g. ``{'time': 10}``. These will always be iterated over.
    concat_input_dims : bool, optional
        If ``True``, the dimension chunks specified in ``input_dims`` will be
        concatenated and stacked into the ``sample`` dimension. The batch index
        will be included as a new level ``input_batch`` in the ``sample``
        coordinate.
        If ``False``, the dimension chunks specified in ``input_dims`` will be
        iterated over.
    preload_batch : bool, optional
        If ``True``, each batch will be loaded into memory before reshaping /
        processing, triggering any dask arrays to be computed.

    Yields
    ------
    ds_slice : ``xarray.Dataset`` or ``xarray.DataArray``
        Slices of the array matching the given batch size specification.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        input_dims: Dict[Hashable, int],
        input_overlap: Dict[Hashable, int] = {},
        batch_dims: Dict[Hashable, int] = {},
        concat_input_dims: bool = False,
        preload_batch: bool = True,
    ):
        super().__init__(
            input_dims=input_dims,
            input_overlap=input_overlap,
            batch_dims=batch_dims,
            concat_input_dims=concat_input_dims,
        )

        self.ds = _as_xarray_dataset(ds)
        # should be a dict
        self.preload_batch = preload_batch

        self._batches: Dict[
            int, Any
        ] = self._gen_batches()  # dict cache for batches
        # in the future, we can make this a lru cache or similar thing (cachey?)

    def __iter__(self) -> Iterator[xr.Dataset]:
        for batch in self._batches.values():
            yield batch

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, idx: int) -> xr.Dataset:

        if not isinstance(idx, int):
            raise NotImplementedError(
                f'{type(self).__name__}.__getitem__ currently requires a single integer key'
            )

        if idx < 0:
            idx = list(self._batches)[idx]

        if idx in self._batches:
            return self._batches[idx]
        else:
            raise IndexError('list index out of range')

    def _gen_batches(self) -> dict:
        # in the future, we will want to do the batch generation lazily
        # going the eager route for now is allowing me to fill out the loader api
        # but it is likely to perform poorly.
        batches = []
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
                    str(dim) + new_dim_suffix for dim in self.input_dims
                ]
                batches.append(_maybe_stack_batch_dims(dsc, new_input_dims))
            else:
                for ds_input in input_generator:
                    batches.append(
                        _maybe_stack_batch_dims(ds_input, list(self.input_dims))
                    )

        return dict(zip(range(len(batches)), batches))

    def _iterate_batch_dims(self, ds):
        return _iterate_through_dataset(ds, self.batch_dims)

    def _iterate_input_dims(self, ds):
        return _iterate_through_dataset(ds, self.input_dims, self.input_overlap)

    def to_zarr(self, path, chunks={'batch': '1Gb'}):
        """
        Store batches into a zarr datastore in `path`. To speed up loading of
        batches it is recommended that the chunking across batches is set close
        to the available RAM on the computere where you are doing ML model
        training
        """
        batch_datasets = list(self)
        # can't call the batch dimension `batch` because Dataset.batch is used
        # for the batch acccessor. Instead we'll call it `batch_number`
        ds_all = xr.concat(batch_datasets, dim='batch_number').reset_index(
            'sample'
        )
        if 'batch' in chunks:
            chunks['batch_number'] = chunks.pop('batch')

        if len(chunks) > 0:
            ds_all = ds_all.chunk(chunks)

        for v in StoredBatchesGenerator.INIT_ARGS_TO_SERIALIZE:
            ds_all.attrs[v] = json.dumps(getattr(self, v))
        ds_all.to_zarr(path)

    @staticmethod
    def from_zarr(path):
        """
        Load a batch generator from the zarr datastore at a given `path`
        """
        return StoredBatchesGenerator(path=path)


class StoredBatchesGenerator(BatchGeneratorBase):
    """
    Create a generator which mimicks the behaviour of BatchGenerator but loads
    the batches from a zarr store that was previously created with
    `BatchGenerator.to_zarr`. Arguments which the original BatchGenerator was
    created with are serialized using json and saved as attributes in the
    zarr-store
    """

    INIT_ARGS_TO_SERIALIZE = [
        'input_dims',
        'input_overlap',
        'batch_dims',
        'concat_input_dims',
    ]

    def __init__(self, path):
        self.ds_batches = xr.open_zarr(path)
        self.path = path

        init_kws = {
            v: json.loads(self.ds_batches.attrs[v])
            for v in self.INIT_ARGS_TO_SERIALIZE
        }
        super().__init__(**init_kws)

    def __iter__(self):
        for batch_id in self.ds_batches.batch_number.values:
            ds_batch = self.ds_batches.sel(batch_number=batch_id)
            # create a MultiIndex like we had before storing the batches
            stacked_coords = [
                d
                for d in ds_batch.coords
                if d not in ['sample', 'batch_number']
            ]
            ds_batch = ds_batch.set_index(sample=stacked_coords)
            yield ds_batch
