"""Classes for iterating through xarray datarrays / datasets in batches."""

import itertools
from typing import Any, Dict, Hashable, Iterator, List, Sequence, Tuple, Union

import xarray as xr

DimSelector = Dict[Hashable, slice]
ConcatBatchSelector = Tuple[DimSelector, List[DimSelector]]
BatchSelector = Union[
    List[ConcatBatchSelector],
    Iterator[DimSelector],
]
BatchSelectors = Union[Dict[int, ConcatBatchSelector], Dict[int, DimSelector]]


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


def _iterate_over_dimensions(
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
        self.input_dims = dict(input_dims)
        self.input_overlap = input_overlap
        self.batch_dims = dict(batch_dims)
        self.concat_input_dims = concat_input_dims
        self.preload_batch = preload_batch
        self._batch_selectors: Dict[
            int, Any
        ] = self._gen_batch_selectors()  # dict cache for batches

    def __iter__(self) -> Iterator[Union[xr.DataArray, xr.Dataset]]:
        for idx in self._batch_selectors:
            yield self[idx]

    def __len__(self) -> int:
        return len(self._batch_selectors)

    def __getitem__(self, idx: int) -> Union[xr.Dataset, xr.DataArray]:

        if not isinstance(idx, int):
            raise NotImplementedError(
                f"{type(self).__name__}.__getitem__ currently requires a single integer key"
            )

        if idx < 0:
            idx = list(self._batch_selectors)[idx]

        if idx in self._batch_selectors:

            if self.concat_input_dims:
                new_dim_suffix = "_input"
                all_dsets: List = []
                batch_dims_selector, input_dims_selectors = self._batch_selectors[idx]
                batch_ds = self.ds.isel(batch_dims_selector)
                if self.preload_batch:
                    batch_ds.load()
                for selector in input_dims_selectors:
                    all_dsets.append(
                        _drop_input_dims(
                            batch_ds.isel(dict(**selector)),
                            self.input_dims,
                            suffix=new_dim_suffix,
                        )
                    )
                dsc = xr.concat(all_dsets, dim="input_batch")
                new_input_dims = [str(dim) + new_dim_suffix for dim in self.input_dims]
                return _maybe_stack_batch_dims(dsc, new_input_dims)
            else:
                batch_ds = self.ds.isel(self._batch_selectors[idx])
                if self.preload_batch:
                    batch_ds.load()
                return _maybe_stack_batch_dims(
                    batch_ds,
                    list(self.input_dims),
                )
        else:
            raise IndexError("list index out of range")

    def _gen_batch_selectors(
        self,
    ) -> BatchSelectors:
        """
        Create batch selectors dict, which can be used to create a batch
        from an xarray data object.
        """
        if self.concat_input_dims:
            batch_dim_selectors = _iterate_over_dimensions(
                self.ds, dims=self.batch_dims
            )
            # TODO: Consider iterator protocol rather than copying to list
            input_dim_selectors = list(
                _iterate_over_dimensions(
                    self.ds, dims=self.input_dims, overlap=self.input_overlap
                )
            )
            batch_selectors: BatchSelector = [
                (selector, input_dim_selectors) for selector in batch_dim_selectors
            ]
        else:
            batch_selectors = _iterate_over_dimensions(
                self.ds,
                dims=dict(**self.batch_dims, **self.input_dims),
                overlap=self.input_overlap,
            )
        return dict(enumerate(batch_selectors))
