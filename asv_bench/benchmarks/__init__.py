import numpy as np


def parameterized(names, params):
    """
    Copied from xarray benchmarks:
    https://github.com/pydata/xarray/blob/main/asv_bench/benchmarks/__init__.py#L9-L15
    """

    def decorator(func):
        func.param_names = names
        func.params = params
        return func

    return decorator


def randn(shape, frac_nan=None, chunks=None, seed=0):
    """
    Copied from xarray benchmarks:
    https://github.com/pydata/xarray/blob/main/asv_bench/benchmarks/__init__.py#L32-L46
    """
    rng = np.random.RandomState(seed)
    if chunks is None:
        x = rng.standard_normal(shape)
    else:
        import dask.array as da

        rng = da.random.RandomState(seed)
        x = rng.standard_normal(shape, chunks=chunks)

    if frac_nan is not None:
        inds = rng.choice(range(x.size), int(x.size * frac_nan))
        x.flat[inds] = np.nan

    return x
