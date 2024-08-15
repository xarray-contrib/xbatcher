from collections.abc import Hashable
from unittest import TestCase

import numpy as np
import xarray as xr

from .generators import BatchGenerator


def _get_non_specified_dims(generator: BatchGenerator) -> dict[Hashable, int]:
    """
    Return all dimensions that are in the input dataset but not ``input_dims``
    or ``batch_dims``.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.

    Returns
    -------
    d : dict
        Dict containing all dimensions in the input dataset that are not
        in the input_dims or batch_dims attributes of the batch generator.
    """
    return {
        dim: length
        for dim, length in generator.ds.sizes.items()
        if generator.input_dims.get(dim) is None
        and generator.batch_dims.get(dim) is None
    }


def _get_non_input_batch_dims(generator: BatchGenerator) -> dict[Hashable, int]:
    """
    Return all dimensions that are in batch_dims but not input_dims.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.

    Returns
    -------
    d : dict
        Dict containing all dimensions in specified in batch_dims that are
        not also in input_dims
    """
    return {
        dim: length
        for dim, length in generator.batch_dims.items()
        if generator.input_dims.get(dim) is None
    }


def _get_duplicate_batch_dims(generator: BatchGenerator) -> dict[Hashable, int]:
    """
    Return all dimensions that are in both batch_dims and input_dims.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.

    Returns
    -------
    d : dict
        Dict containing all dimensions duplicated between batch_dims and input_dims.
    """
    return {
        dim: length
        for dim, length in generator.batch_dims.items()
        if generator.input_dims.get(dim) is not None
    }


def _get_sample_length(
    *,
    generator: BatchGenerator,
    non_specified_ds_dims: dict[Hashable, int],
    non_input_batch_dims: dict[Hashable, int],
) -> int:
    """
    Return the expected length of the sample dimension.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.
    non_specified_ds_dics : dict
        Dict containing all dimensions in the input dataset that are not
        in the input_dims or batch_dims attributes of the batch generator.
    non_input_batch_dims : dict
        Dict containing all dimensions in specified in batch_dims that are
        not also in input_dims

    Returns
    -------
    s : int
        Expected length of the sample dimension
    """
    if generator.concat_input_dims:
        batch_concat_dims = [
            (
                generator.batch_dims.get(dim) // length
                if generator.batch_dims.get(dim)
                else generator.ds.sizes.get(dim) // length
            )
            for dim, length in generator.input_dims.items()
        ]
    else:
        batch_concat_dims = []
    return int(
        np.prod(list(non_specified_ds_dims.values()))
        * np.prod(list(non_input_batch_dims.values()))
        * np.prod(batch_concat_dims)
    )


def get_batch_dimensions(generator: BatchGenerator) -> dict[Hashable, int]:
    """
    Return the expected batch dimensions based on the ``input_dims``,
    ``batch_dims``, and ``concat_input_dims`` attributes of the batch
    generator.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.

    Returns
    -------
    d : dict
        Dict containing the expected dimensions for batches returned by the
        batch generator.
    """
    # dimensions that are in the input dataset but not input_dims or batch_dims
    non_specified_ds_dims = _get_non_specified_dims(generator)
    # dimensions that are in batch_dims but not input_dims
    non_input_batch_dims = _get_non_input_batch_dims(generator)
    expected_sample_length = _get_sample_length(
        generator=generator,
        non_specified_ds_dims=non_specified_ds_dims,
        non_input_batch_dims=non_input_batch_dims,
    )
    # input_dims stay the same, possibly with a new suffix
    expected_dims = {
        f'{k}_input' if generator.concat_input_dims else k: v
        for k, v in generator.input_dims.items()
    }
    # Add a sample dimension if there's anything to get stacked
    if (
        generator.concat_input_dims
        and (len(generator.ds.sizes) - len(generator.input_dims)) == 0
    ):
        expected_dims = {**{'input_batch': expected_sample_length}, **expected_dims}
    elif (
        generator.concat_input_dims
        or (len(generator.ds.sizes) - len(generator.input_dims)) > 1
    ):
        expected_dims = {**{'sample': expected_sample_length}, **expected_dims}
    else:
        expected_dims = dict(
            **non_specified_ds_dims,
            **non_input_batch_dims,
            **expected_dims,
        )
    return expected_dims


def validate_batch_dimensions(
    *, expected_dims: dict[Hashable, int], batch: xr.Dataset | xr.DataArray
) -> None:
    """
    Raises an AssertionError if the shape and dimensions of a batch do not
    match expected_dims.

    Parameters
    ----------
    expected_dims : Dict
        Dict containing the expected dimensions for batches.
    batch : xarray.Dataset or xarray.DataArray
        The xarray data object returned by the batch generator.
    """

    # Check the names and lengths of the dimensions are equal
    TestCase().assertDictEqual(
        expected_dims, dict(batch.sizes), msg='Dimension names and/or lengths differ'
    )
    # Check the dimension order is equal
    for var in batch.data_vars:
        TestCase().assertEqual(
            tuple(expected_dims.values()),
            batch[var].shape,
            msg=f'Order differs for dimensions of: {expected_dims}',
        )


def _get_nbatches_from_input_dims(generator: BatchGenerator) -> int:
    """
    Calculate the number of batches expected based on ``input_dims`` and
    ``input_overlap``.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.

    Returns
    -------
    s : int
        Number of batches expected given ``input_dims`` and ``input_overlap``.
    """
    nbatches_from_input_dims = np.prod(
        [
            generator.ds.sizes[dim] // length
            for dim, length in generator.input_dims.items()
            if generator.input_overlap.get(dim) is None
            and generator.batch_dims.get(dim) is None
        ]
    )
    if generator.input_overlap:
        nbatches_from_input_overlap = np.prod(
            [
                (generator.ds.sizes[dim] - overlap)
                // (generator.input_dims[dim] - overlap)
                for dim, overlap in generator.input_overlap.items()
            ]
        )
        return int(nbatches_from_input_overlap * nbatches_from_input_dims)
    else:
        return int(nbatches_from_input_dims)


def validate_generator_length(generator: BatchGenerator) -> None:
    """
    Raises an AssertionError if the generator length does not match
    expectations based on the input Dataset and ``input_dims``.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.
    """
    non_input_batch_dims = _get_non_input_batch_dims(generator)
    duplicate_batch_dims = _get_duplicate_batch_dims(generator)
    nbatches_from_unique_batch_dims = np.prod(
        [
            generator.ds.sizes[dim] // length
            for dim, length in non_input_batch_dims.items()
        ]
    )
    nbatches_from_duplicate_batch_dims = np.prod(
        [
            generator.ds.sizes[dim] // length
            for dim, length in duplicate_batch_dims.items()
        ]
    )
    if generator.concat_input_dims:
        expected_length = int(
            nbatches_from_unique_batch_dims * nbatches_from_duplicate_batch_dims
        )
    else:
        nbatches_from_input_dims = _get_nbatches_from_input_dims(generator)
        expected_length = int(
            nbatches_from_unique_batch_dims
            * nbatches_from_duplicate_batch_dims
            * nbatches_from_input_dims
        )
    TestCase().assertEqual(
        expected_length,
        len(generator),
        msg='Batch generator length differs',
    )
