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
