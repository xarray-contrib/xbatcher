from unittest import TestCase

import numpy as np


def _nbatches_from_batch_dims(generator):
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
        k: v
        for k, v in generator.ds.dims.items()
        if (generator.input_dims.get(k) is None and generator.batch_dims.get(k) is None)
    }


def _get_non_input_batch_dims(generator):
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
        k: v
        for k, v in generator.batch_dims.items()
        if (generator.input_dims.get(k) is None)
    }


def _get_sample_length(*, generator, non_specified_ds_dims, non_input_batch_dims):
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
            generator.ds.dims.get(k)
            // np.nanmax([v, generator.batch_dims.get(k, np.nan)])
            for k, v in generator.input_dims.items()
        ]
    else:
        batch_concat_dims = []
    return int(
        np.product(list(non_specified_ds_dims.values()))
        * np.product(list(non_input_batch_dims.values()))
        * np.product(batch_concat_dims)
    )


def validate_batch_dimensions(*, generator, batch):
    """
    Raises an AssertionError if the shape and dimensions of a batch are not
    as expected based on the ``input_dims``, ``batch_dims``, and
    ``concat_input_dims`` attributes of the batch generator.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object used to return the batch.
    batch : xarray.Dataset or xarray.DataArray
        The xarray data object returned by the batch generator.
    """
    # dimensions that are in the input dataset but not input_dims or batch_dims
    non_specified_ds_dims = _nbatches_from_batch_dims(generator)
    # dimensions that are in batch_dims but not input_dims
    non_input_batch_dims = _get_non_input_batch_dims(generator)
    expected_sample_length = _get_sample_length(
        generator=generator,
        non_specified_ds_dims=non_specified_ds_dims,
        non_input_batch_dims=non_input_batch_dims,
    )
    suffix = "_input" if generator.concat_input_dims else ""
    # input_dims stay the same, possibly with a new suffix
    expected_dims = {
        f"{k}{suffix}": generator.input_dims.get(k)
        for k in generator.ds.dims.keys()
        if generator.input_dims.get(k) is not None
    }
    # Add a sample dimension if there's anything to get stacked
    if generator.concat_input_dims and len(non_specified_ds_dims) < 1:
        expected_dims = {**{"input_batch": expected_sample_length}, **expected_dims}
    elif (
        generator.concat_input_dims
        or generator.batch_dims
        or len(non_specified_ds_dims) > 1
    ):
        expected_dims = {**{"sample": expected_sample_length}, **expected_dims}
    else:
        expected_dims = {**non_specified_ds_dims, **expected_dims}
    # Check the names and lengths of the dimensions are equal
    TestCase().assertDictEqual(
        expected_dims, batch.dims.mapping, msg="Dimension names and/or lengths differ"
    )
    # Check the dimension order is equal
    for var in batch.data_vars:
        TestCase().assertEqual(
            tuple(expected_dims.values()),
            batch[var].shape,
            msg=f"Order differs for dimensions of: {expected_dims}",
        )


def _get_nbatches_from_input_dims(generator):
    """
    Calculate the number of batches expected based on ``input_dims`` and
    ``input_overlap``.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.

    Returns
    -------
    s : float
        Number of batches expected given ``input_dims`` and ``input_overlap``.
    """
    nbatches_from_input_dims = np.product(
        [
            generator.ds.dims[k] // generator.input_dims[k]
            for k in generator.input_dims.keys()
            if generator.input_overlap.get(k) is None
        ]
    )
    if generator.input_overlap:
        nbatches_from_input_overlap = np.product(
            [
                (generator.ds.dims[k] - generator.input_overlap[k])
                // (generator.input_dims[k] - generator.input_overlap[k])
                for k in generator.input_overlap
            ]
        )
        return nbatches_from_input_overlap * nbatches_from_input_dims
    else:
        return nbatches_from_input_dims


def validate_generator_length(generator):
    """
    Raises an AssertionError if the generator length does not match
    expectations based on the input Dataset and ``input_dims``.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.
    """
    non_input_batch_dims = _get_non_input_batch_dims(generator)
    nbatches_from_batch_dims = np.product(
        [
            generator.ds.dims[k] // non_input_batch_dims[k]
            for k in non_input_batch_dims.keys()
        ]
    )
    if generator.concat_input_dims:
        expected_length = int(nbatches_from_batch_dims)
    else:
        nbatches_from_input_dims = _get_nbatches_from_input_dims(generator)
        expected_length = int(nbatches_from_batch_dims * nbatches_from_input_dims)
    TestCase().assertEqual(
        expected_length,
        len(generator),
        msg="Batch generator length differs",
    )
