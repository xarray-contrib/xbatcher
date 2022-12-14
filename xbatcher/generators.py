"""Classes for iterating through xarray datarrays / datasets in batches."""

import itertools
from dataclasses import dataclass
from operator import itemgetter
from typing import Dict, Hashable, Iterator, List, Sequence, Union

import xarray as xr

BatchSelector = List[Dict[Hashable, slice]]
BatchSelectorSet = Dict[int, BatchSelector]


@dataclass
class BatchSchema:
    """
    A representation of the indices and stacking/transposing parameters needed
    to generator batches from Xarray Datasets and DataArrays using
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
        ds: Union[xr.Dataset, xr.DataArray],
        input_dims: Dict[Hashable, int],
        input_overlap: Dict[Hashable, int] = {},
        batch_dims: Dict[Hashable, int] = {},
        concat_input_bins: bool = True,
        preload_batch: bool = True,
    ):
        self.input_dims = dict(input_dims)
        self.input_overlap = input_overlap
        self.batch_dims = dict(batch_dims)
        self.concat_input_dims = concat_input_bins
        self.preload_batch = preload_batch
        self.selectors: BatchSelectorSet = self._gen_batch_selectors(ds)

    def _gen_batch_selectors(self, ds) -> BatchSelectorSet:
        """
        Create batch selectors dict, which can be used to create a batch
        from an xarray data object.
        """
        # Separate batch_dims that are/are not also included in input_dims
        self._duplicate_batch_dims = {
            dim: length
            for dim, length in self.batch_dims.items()
            if self.input_dims.get(dim) is not None
        }
        self._unique_batch_dims = {
            dim: length
            for dim, length in self.batch_dims.items()
            if self.input_dims.get(dim) is None
        }
        # Create an iterator that returns an object usable for .isel in xarray
        patch_selectors = self._gen_patch_selectors(ds)
        # Create the Dict containing batch selectors
        if self.concat_input_dims:  # Combine the patches into batches
            batch_selectors = self._combine_patches_into_batch(ds, patch_selectors)
            return dict(enumerate(batch_selectors))
        else:  # Each patch gets its own batch
            return {ind: [value] for ind, value in enumerate(patch_selectors)}

    def _gen_patch_selectors(self, ds) -> Iterator[Dict[Hashable, slice]]:
        """
        Create an iterator that can be used to index an Xarray Dataset/DataArray.
        """
        if self._duplicate_batch_dims and not self.concat_input_dims:
            raise UserWarning(
                f"""
                The following dimensions were included in both ``input_dims``
                and ``batch_dims``. Since ``concat_input_dims`` is ``False``,
                these dimensions will not impact batch generation: {self._duplicate_batch_dims}
                """
            )
        # Generate the slices by iterating over batch_dims and input_dims
        all_slices = _iterate_through_dimensions(
            ds,
            dims=dict(**self._unique_batch_dims, **self.input_dims),
            overlap=self.input_overlap,
        )
        return all_slices

    def _combine_patches_into_batch(
        self, ds, patch_selectors
    ) -> List[List[Dict[Hashable, slice]]]:
        """
        Combine the patch selectors to form a batch
        """
        # Check that patches are only combined with concat_input_dims
        if not self.concat_input_dims:
            raise AssertionError(
                "Patches should only be combined into batches when ``concat_input_dims`` is ``True``"
            )
        # If ``batch_dims`` isn't used, all patches will be included in a single batch
        if not self.batch_dims:
            batch_selectors = [list(patch_selectors)]
        elif self._duplicate_batch_dims:
            raise NotImplementedError("Not Implemented")
        # Group patches based on the unique slices for dimensions in ``batch_dims``
        else:
            batch_selectors = [
                list(value)
                for _, value in itertools.groupby(
                    patch_selectors, key=itemgetter(*self.batch_dims)
                )
            ]
        # Group patches based on the unique dimensions in ``batch_dims``
        return batch_selectors


def _gen_slices(*, dim_size: int, slice_size: int, overlap: int = 0) -> List[slice]:
    # return a list of slices to chop up a single dimension
    if overlap >= slice_size:
        raise ValueError(
            "input overlap must be less than the input sample length, but "
            f"the input sample length is {slice_size} and the overlap is {overlap}"
        )
    slices = []
    stride = slice_size - overlap
    for start in range(0, dim_size, stride):
        end = start + slice_size
        if end <= dim_size:
            slices.append(slice(start, end))
    return slices


def _iterate_through_dimensions(
    ds: Union[xr.Dataset, xr.DataArray],
    *,
    dims: Dict[Hashable, int],
    overlap: Dict[Hashable, int] = {},
) -> Iterator[Dict[Hashable, slice]]:
    dim_slices = []
    for dim in dims:
        dim_size = ds.sizes[dim]
        slice_size = dims[dim]
        slice_overlap = overlap.get(dim, 0)
        if slice_size > dim_size:
            raise ValueError(
                "input sample length must be less than or equal to the "
                f"dimension length, but the sample length of {slice_size} "
                f"is greater than the dimension length of {dim_size} "
                f"for {dim}"
            )
        dim_slices.append(
            _gen_slices(dim_size=dim_size, slice_size=slice_size, overlap=slice_overlap)
        )
    for slices in itertools.product(*dim_slices):
        selector = dict(zip(dims, slices))
        yield selector


def _drop_input_dims(
    ds: Union[xr.Dataset, xr.DataArray],
    input_dims: Dict[Hashable, int],
    suffix: str = "_input",
) -> Union[xr.Dataset, xr.DataArray]:
    # remove input_dims coordinates from datasets, rename the dimensions
    # then put intput_dims back in as coordinates
    out = ds.copy()
    for dim in input_dims.keys():
        newdim = f"{dim}{suffix}"
        out = out.rename({dim: newdim})
        # extra steps needed if there is a coordinate
        if newdim in out:
            out = out.drop_vars(newdim)
            out.coords[dim] = newdim, ds[dim].data, ds[dim].attrs
    return out


def _maybe_stack_batch_dims(
    ds: Union[xr.Dataset, xr.DataArray],
    input_dims: Sequence[Hashable],
) -> Union[xr.Dataset, xr.DataArray]:
    batch_dims = [d for d in ds.sizes if d not in input_dims]
    if len(batch_dims) < 2:
        return ds
    ds_stack = ds.stack(sample=batch_dims)
    # ensure correct order
    dim_order = ("sample",) + tuple(input_dims)
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
        ds: Union[xr.Dataset, xr.DataArray],
        input_dims: Dict[Hashable, int],
        input_overlap: Dict[Hashable, int] = {},
        batch_dims: Dict[Hashable, int] = {},
        concat_input_dims: bool = False,
        preload_batch: bool = True,
    ):

        self.ds = ds
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

    def __iter__(self) -> Iterator[Union[xr.DataArray, xr.Dataset]]:
        for idx in self._batch_selectors.selectors:
            yield self[idx]

    def __len__(self) -> int:
        return len(self._batch_selectors.selectors)

    def __getitem__(self, idx: int) -> Union[xr.Dataset, xr.DataArray]:

        if not isinstance(idx, int):
            raise NotImplementedError(
                f"{type(self).__name__}.__getitem__ currently requires a single integer key"
            )

        if idx < 0:
            idx = list(self._batch_selectors.selectors)[idx]

        if idx in self._batch_selectors.selectors:

            if self.concat_input_dims:
                new_dim_suffix = "_input"
                all_dsets: List = []
                for selector in self._batch_selectors.selectors[idx]:
                    batch_ds = self.ds.isel(selector)
                    if self.preload_batch:
                        batch_ds.load()
                    all_dsets.append(
                        _drop_input_dims(
                            batch_ds,
                            self.input_dims,
                            suffix=new_dim_suffix,
                        )
                    )
                dsc = xr.concat(all_dsets, dim="input_batch")
                new_input_dims = [str(dim) + new_dim_suffix for dim in self.input_dims]
                return _maybe_stack_batch_dims(dsc, new_input_dims)
            else:
                batch_ds = self.ds.isel(self._batch_selectors.selectors[idx][0])
                if self.preload_batch:
                    batch_ds.load()
                return _maybe_stack_batch_dims(
                    batch_ds,
                    list(self.input_dims),
                )
        else:
            raise IndexError("list index out of range")
