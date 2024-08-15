"""Classes for iterating through xarray datarrays / datasets in batches."""

import itertools
import json
import warnings
from collections.abc import Callable, Hashable, Iterator, Sequence
from operator import itemgetter
from typing import Any

import numpy as np
import xarray as xr

PatchGenerator = Iterator[dict[Hashable, slice]]
BatchSelector = list[dict[Hashable, slice]]
BatchSelectorSet = dict[int, BatchSelector]


class BatchSchema:
    """
    A representation of the indices and stacking/transposing parameters needed
    to generator batches from Xarray DataArrays and Datasets using
    xbatcher.BatchGenerator.

    Parameters
    ----------
    ds : ``xarray.Dataset`` or ``xarray.DataArray``
        The data to iterate over. Unlike for the BatchGenerator, the data is
        not retained as a class attribute for the BatchSchema.
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

    Notes
    -----
    The BatchSchema is experimental and subject to change without notice.
    """

    def __init__(
        self,
        ds: xr.Dataset | xr.DataArray,
        input_dims: dict[Hashable, int],
        input_overlap: dict[Hashable, int] | None = None,
        batch_dims: dict[Hashable, int] | None = None,
        concat_input_bins: bool = True,
        preload_batch: bool = True,
    ):
        if input_overlap is None:
            input_overlap = {}
        if batch_dims is None:
            batch_dims = {}
        self.input_dims = dict(input_dims)
        self.input_overlap = input_overlap
        self.batch_dims = dict(batch_dims)
        self.concat_input_dims = concat_input_bins
        self.preload_batch = preload_batch
        # Store helpful information based on arguments
        self._duplicate_batch_dims: dict[Hashable, int] = {
            dim: length
            for dim, length in self.batch_dims.items()
            if self.input_dims.get(dim) is not None
        }
        self._unique_batch_dims: dict[Hashable, int] = {
            dim: length
            for dim, length in self.batch_dims.items()
            if self.input_dims.get(dim) is None
        }
        self._input_stride: dict[Hashable, int] = {
            dim: length - self.input_overlap.get(dim, 0)
            for dim, length in self.input_dims.items()
        }
        self._all_sliced_dims: dict[Hashable, int] = dict(
            **self._unique_batch_dims, **self.input_dims
        )
        self.selectors: BatchSelectorSet = self._gen_batch_selectors(ds)

    def _gen_batch_selectors(self, ds: xr.DataArray | xr.Dataset) -> BatchSelectorSet:
        """
        Create batch selectors dict, which can be used to create a batch
        from an Xarray data object.
        """
        # Create an iterator that returns an object usable for .isel in xarray
        patch_selectors = self._gen_patch_selectors(ds)
        # Create the Dict containing batch selectors
        if self.concat_input_dims:  # Combine the patches into batches
            return self._combine_patches_into_batch(ds, patch_selectors)
        else:  # Each patch gets its own batch
            return {ind: [value] for ind, value in enumerate(patch_selectors)}

    def _gen_patch_selectors(self, ds: xr.DataArray | xr.Dataset) -> PatchGenerator:
        """
        Create an iterator that can be used to index an Xarray Dataset/DataArray.
        """
        if self._duplicate_batch_dims and not self.concat_input_dims:
            warnings.warn(
                'The following dimensions were included in both ``input_dims`` '
                'and ``batch_dims``. Since ``concat_input_dims`` is ``False``, '
                f'these dimensions will not impact batch generation: {self._duplicate_batch_dims}'
            )
        # Generate the slices by iterating over batch_dims and input_dims
        all_slices = _iterate_through_dimensions(
            ds,
            dims=self._all_sliced_dims,
            overlap=self.input_overlap,
        )
        return all_slices

    def _combine_patches_into_batch(
        self, ds: xr.DataArray | xr.Dataset, patch_selectors: PatchGenerator
    ) -> BatchSelectorSet:
        """
        Combine the patch selectors to form a batch
        """
        # Check that patches are only combined with concat_input_dims
        if not self.concat_input_dims:
            raise AssertionError(
                'Patches should only be combined into batches when ``concat_input_dims`` is ``True``'
            )
        if not self.batch_dims:
            return self._combine_patches_into_one_batch(patch_selectors)
        elif self._duplicate_batch_dims:
            return self._combine_patches_grouped_by_input_and_batch_dims(
                ds=ds, patch_selectors=patch_selectors
            )
        else:
            return self._combine_patches_grouped_by_batch_dims(patch_selectors)

    def _combine_patches_into_one_batch(
        self, patch_selectors: PatchGenerator
    ) -> BatchSelectorSet:
        """
        Group all patches into a single batch
        """
        return dict(enumerate([list(patch_selectors)]))

    def _combine_patches_grouped_by_batch_dims(
        self, patch_selectors: PatchGenerator
    ) -> BatchSelectorSet:
        """
        Group patches based on the unique slices for dimensions in ``batch_dims``
        """
        batch_selectors = [
            list(value)
            for _, value in itertools.groupby(
                patch_selectors, key=itemgetter(*self.batch_dims)
            )
        ]
        return dict(enumerate(batch_selectors))

    def _combine_patches_grouped_by_input_and_batch_dims(
        self, ds: xr.DataArray | xr.Dataset, patch_selectors: PatchGenerator
    ) -> BatchSelectorSet:
        """
        Combine patches with multiple slices along ``batch_dims`` grouped into
        each patch. Required when a dimension is duplicated between ``batch_dims``
        and ``input_dims``.
        """
        self._gen_patch_numbers(ds)
        self._gen_batch_numbers(ds)
        batch_id_per_patch = self._get_batch_multi_index_per_patch()
        patch_in_range = self._get_batch_in_range_per_batch(
            batch_multi_index=batch_id_per_patch
        )
        batch_id_per_patch = self._ravel_batch_multi_index(batch_id_per_patch)
        batch_selectors = self._gen_empty_batch_selectors()
        for i, patch in enumerate(patch_selectors):
            if patch_in_range[i]:
                batch_selectors[batch_id_per_patch[i]].append(patch)
        return batch_selectors

    def _gen_empty_batch_selectors(self) -> BatchSelectorSet:
        """
        Create an empty batch selector set that can be populated by appending
        patches to each batch.
        """
        n_batches = np.prod(list(self._n_batches_per_dim.values()))
        return {k: [] for k in range(n_batches)}

    def _gen_patch_numbers(self, ds: xr.DataArray | xr.Dataset):
        """
        Calculate the number of patches per dimension and the number of patches
        in each batch per dimension.
        """
        self._n_patches_per_batch: dict[Hashable, int] = {
            dim: int(np.ceil(length / self._input_stride.get(dim, length)))
            for dim, length in self.batch_dims.items()
        }
        self._n_patches_per_dim: dict[Hashable, int] = {
            dim: int(
                (ds.sizes[dim] - self.input_overlap.get(dim, 0))
                // (length - self.input_overlap.get(dim, 0))
            )
            for dim, length in self._all_sliced_dims.items()
        }

    def _gen_batch_numbers(self, ds: xr.DataArray | xr.Dataset):
        """
        Calculate the number of batches per dimension
        """
        self._n_batches_per_dim: dict[Hashable, int] = {
            dim: int(ds.sizes[dim] // self.batch_dims.get(dim, ds.sizes[dim]))
            for dim in self._all_sliced_dims.keys()
        }

    def _get_batch_multi_index_per_patch(self):
        """
        Calculate the batch multi-index for each patch
        """
        batch_id_per_dim: dict[Hashable, Any] = {
            dim: np.floor(
                np.arange(0, n_patches)
                / self._n_patches_per_batch.get(dim, n_patches + 1)
            ).astype(np.int64)
            for dim, n_patches in self._n_patches_per_dim.items()
        }
        batch_id_per_patch = np.array(
            list(itertools.product(*batch_id_per_dim.values()))
        ).transpose()
        return batch_id_per_patch

    def _ravel_batch_multi_index(self, batch_multi_index):
        """
        Convert the batch multi-index to a flat index for each patch
        """
        return np.ravel_multi_index(
            multi_index=batch_multi_index,
            dims=tuple(self._n_batches_per_dim.values()),
            mode='clip',
        )

    def _get_batch_in_range_per_batch(self, batch_multi_index):
        """
        Determine whether each patch is contained within any of the batches.
        """
        batch_id_maximum = np.fromiter(self._n_batches_per_dim.values(), dtype=int)
        batch_id_maximum = np.pad(
            batch_id_maximum,
            (0, (len(self._n_patches_per_dim) - len(self._n_batches_per_dim))),
            constant_values=(1),
        )
        batch_id_maximum = batch_id_maximum[:, np.newaxis]
        batch_in_range_per_patch = np.all(batch_multi_index < batch_id_maximum, axis=0)
        return batch_in_range_per_patch

    def to_json(self):
        """
        Dump the BatchSchema properties to a JSON file.

        Returns
        ----------
        out_json: str
            The JSON representation of the BatchSchema
        """
        out_dict = {}
        out_dict['input_dims'] = self.input_dims
        out_dict['input_overlap'] = self.input_overlap
        out_dict['batch_dims'] = self.batch_dims
        out_dict['concat_input_dims'] = self.input_dims
        out_dict['preload_batch'] = self.preload_batch
        batch_selector_dict = {}
        for i in self.selectors.keys():
            batch_selector_dict[i] = self.selectors[i]
            for member in batch_selector_dict[i]:
                out_member_dict = {}
                member_keys = [x for x in member.keys()]
                for member_key in member_keys:
                    out_member_dict[member_key] = {
                        'start': member[member_key].start,
                        'stop': member[member_key].stop,
                        'step': member[member_key].step,
                    }
        out_dict['selector'] = out_member_dict
        return json.dumps(out_dict)

    def to_file(self, out_file_name: str):
        """
        Dumps the JSON representation of the BatchSchema object to a file.

        Parameters
        ----------
        out_file_name: str
            The path to the json file to write to.
        """
        out_json = self.to_json()
        with open(out_file_name, mode='w') as out_file:
            out_file.write(out_json)


def _gen_slices(*, dim_size: int, slice_size: int, overlap: int = 0) -> list[slice]:
    # return a list of slices to chop up a single dimension
    if overlap >= slice_size:
        raise ValueError(
            'input overlap must be less than the input sample length, but '
            f'the input sample length is {slice_size} and the overlap is {overlap}'
        )
    slices = []
    stride = slice_size - overlap
    for start in range(0, dim_size, stride):
        end = start + slice_size
        if end <= dim_size:
            slices.append(slice(start, end))
    return slices


def _iterate_through_dimensions(
    ds: xr.Dataset | xr.DataArray,
    *,
    dims: dict[Hashable, int],
    overlap: dict[Hashable, int] = {},
) -> Iterator[dict[Hashable, slice]]:
    dim_slices = []
    for dim in dims:
        dim_size = ds.sizes[dim]
        slice_size = dims[dim]
        slice_overlap = overlap.get(dim, 0)
        if slice_size > dim_size:
            raise ValueError(
                'input sample length must be less than or equal to the '
                f'dimension length, but the sample length of {slice_size} '
                f'is greater than the dimension length of {dim_size} '
                f'for {dim}'
            )
        dim_slices.append(
            _gen_slices(dim_size=dim_size, slice_size=slice_size, overlap=slice_overlap)
        )
    for slices in itertools.product(*dim_slices):
        selector = dict(zip(dims, slices))
        yield selector


def _drop_input_dims(
    ds: xr.Dataset | xr.DataArray,
    input_dims: dict[Hashable, int],
    suffix: str = '_input',
) -> xr.Dataset | xr.DataArray:
    # remove input_dims coordinates from datasets, rename the dimensions
    # then put intput_dims back in as coordinates
    out = ds.copy()
    for dim in input_dims.keys():
        newdim = f'{dim}{suffix}'
        out = out.rename({dim: newdim})
        # extra steps needed if there is a coordinate
        if newdim in out:
            out = out.drop_vars(newdim)
            out.coords[dim] = newdim, ds[dim].data, ds[dim].attrs
    return out


def _maybe_stack_batch_dims(
    ds: xr.Dataset | xr.DataArray,
    input_dims: Sequence[Hashable],
) -> xr.Dataset | xr.DataArray:
    batch_dims = [d for d in ds.sizes if d not in input_dims]
    if len(batch_dims) < 2:
        return ds
    ds_stack = ds.stack(sample=batch_dims)
    # ensure correct order
    dim_order = ('sample',) + tuple(input_dims)
    return ds_stack.transpose(*dim_order)


class BatchGenerator:
    """Create generator for iterating through Xarray DataArrays / Datasets in
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
    cache : dict, optional
        Dict-like object to cache batches in (e.g., Zarr DirectoryStore). Note:
        The caching API is experimental and subject to change.
    cache_preprocess: callable, optional
        A function to apply to batches prior to caching.
        Note: The caching API is experimental and subject to change.

    Yields
    ------
    ds_slice : ``xarray.Dataset`` or ``xarray.DataArray``
        Slices of the array matching the given batch size specification.
    """

    def __init__(
        self,
        ds: xr.Dataset | xr.DataArray,
        input_dims: dict[Hashable, int],
        input_overlap: dict[Hashable, int] = {},
        batch_dims: dict[Hashable, int] = {},
        concat_input_dims: bool = False,
        preload_batch: bool = True,
        cache: dict[str, Any] | None = None,
        cache_preprocess: Callable | None = None,
    ):
        self.ds = ds
        self.cache = cache
        self.cache_preprocess = cache_preprocess

        self._batch_selectors: BatchSchema = BatchSchema(
            ds,
            input_dims=input_dims,
            input_overlap=input_overlap,
            batch_dims=batch_dims,
            concat_input_bins=concat_input_dims,
            preload_batch=preload_batch,
        )

    @property
    def input_dims(self):
        return self._batch_selectors.input_dims

    @property
    def input_overlap(self):
        return self._batch_selectors.input_overlap

    @property
    def batch_dims(self):
        return self._batch_selectors.batch_dims

    @property
    def concat_input_dims(self):
        return self._batch_selectors.concat_input_dims

    @property
    def preload_batch(self):
        return self._batch_selectors.preload_batch

    def __iter__(self) -> Iterator[xr.DataArray | xr.Dataset]:
        for idx in self._batch_selectors.selectors:
            yield self[idx]

    def __len__(self) -> int:
        return len(self._batch_selectors.selectors)

    def __getitem__(self, idx: int) -> xr.Dataset | xr.DataArray:
        if not isinstance(idx, int):
            raise NotImplementedError(
                f'{type(self).__name__}.__getitem__ currently requires a single integer key'
            )

        if idx < 0:
            idx = list(self._batch_selectors.selectors)[idx]

        if self.cache and self._batch_in_cache(idx):
            return self._get_cached_batch(idx)

        if idx in self._batch_selectors.selectors:
            if self.concat_input_dims:
                new_dim_suffix = '_input'
                all_dsets: list = []
                batch_selector = {}
                for dim in self._batch_selectors.batch_dims.keys():
                    starts = [
                        x[dim].start for x in self._batch_selectors.selectors[idx]
                    ]
                    stops = [x[dim].stop for x in self._batch_selectors.selectors[idx]]
                    batch_selector[dim] = slice(min(starts), max(stops))
                batch_ds = self.ds.isel(batch_selector)
                if self.preload_batch:
                    batch_ds.load()
                for selector in self._batch_selectors.selectors[idx]:
                    patch_ds = self.ds.isel(selector)
                    all_dsets.append(
                        _drop_input_dims(
                            patch_ds,
                            self.input_dims,
                            suffix=new_dim_suffix,
                        )
                    )
                dsc = xr.concat(all_dsets, dim='input_batch')
                new_input_dims = [str(dim) + new_dim_suffix for dim in self.input_dims]
                batch = _maybe_stack_batch_dims(dsc, new_input_dims)
            else:
                batch_ds = self.ds.isel(self._batch_selectors.selectors[idx][0])
                if self.preload_batch:
                    batch_ds.load()
                batch = _maybe_stack_batch_dims(
                    batch_ds,
                    list(self.input_dims),
                )
        else:
            raise IndexError('list index out of range')

        if self.cache is not None and self.cache_preprocess is not None:
            batch = self.cache_preprocess(batch)
        if self.cache is not None:
            self._cache_batch(idx, batch)

        return batch

    def _batch_in_cache(self, idx: int) -> bool:
        return self.cache is not None and f'{idx}/.zgroup' in self.cache

    def _cache_batch(self, idx: int, batch: xr.Dataset | xr.DataArray) -> None:
        batch.to_zarr(self.cache, group=str(idx), mode='a')

    def _get_cached_batch(self, idx: int) -> xr.Dataset:
        ds = xr.open_zarr(self.cache, group=str(idx))
        if self.preload_batch:
            ds = ds.load()
        return ds
