import datatree.testing as dtt
import numpy as np
import pytest
import xarray as xr
from datatree import DataTree

from xbatcher import datatree_slice_generator


@pytest.fixture(scope="module")
def sample_datatree() -> DataTree:
    """
    Sample multi-resolution DataTree for testing.

    DataTree('None', parent=None)
    ├── DataTree('grid10m')
    │       Dimensions:  (y: 4, x: 6)
    │       Coordinates:
    │         * y        (y) float64 40.0 30.0 20.0 10.0
    │         * x        (x) float64 100.0 110.0 120.0 130.0 140.0 150.0
    │       Data variables:
    │           grid10m  (y, x) int64 10 11 12 13 14 15 16 17 18 ... 26 27 28 29 30 31 32 33
    └── DataTree('grid20m')
            Dimensions:  (y: 2, x: 3)
            Coordinates:
              * y        (y) float64 35.0 15.0
              * x        (x) float64 105.0 125.0 145.0
            Data variables:
                grid20m  (y, x) int64 0 1 2 3 4 5
    """
    grid20m = xr.DataArray(
        data=np.arange(0, 6).reshape(2, 3),
        dims=("y", "x"),
        coords={"y": np.linspace(35.0, 15.0, num=2), "x": np.linspace(105, 145, num=3)},
        name="grid20m",
    )
    grid10m = xr.DataArray(
        data=np.arange(10, 34).reshape(4, 6),
        dims=("y", "x"),
        coords={"y": np.linspace(40.0, 10.0, num=4), "x": np.linspace(100, 150, num=6)},
        name="grid10m",
    )
    dt = DataTree.from_dict(d={"grid10m": grid10m, "grid20m": grid20m})
    return dt


@pytest.fixture(scope="module")
def expected_datatree() -> DataTree:
    """ """
    expected_datatree = DataTree.from_dict(
        d={
            "grid10m": xr.DataArray(
                data=[[26, 27], [32, 33]],
                dims=("y", "x"),
                coords={"y": [20.0, 10.0], "x": [140.0, 150.0]},
                name="grid10m",
            ),
            "grid20m": xr.DataArray(
                data=[[5]],
                dims=("y", "x"),
                coords={"y": [15.0], "x": [145.0]},
                name="grid20m",
            ),
        }
    )
    return expected_datatree


def test_datatree(sample_datatree, expected_datatree):
    """
    Test slicing through a multi-resolution DataTree.
    """
    generator = datatree_slice_generator(
        data_obj=sample_datatree, dim_strides={"y": -20, "x": 20}, ref_node="grid20m"
    )
    for i, chip in enumerate(generator):
        pass

    assert i + 1 == 6  # number of chips
    dtt.assert_identical(a=chip, b=expected_datatree)  # check returned DataTree
