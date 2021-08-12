# type: ignore
import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace, tmpdir):
    import numpy as np

    # always seed numpy.random to make the examples deterministic
    np.random.seed(0)
