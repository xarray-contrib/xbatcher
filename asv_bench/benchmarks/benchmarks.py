import numpy as np
import torch
import xarray as xr

from xbatcher import BatchGenerator
from xbatcher.loaders.torch import IterableDataset, MapDataset

from . import parameterized


class Base:
    def setup(self, *args, **kwargs):
        shape = (10, 50, 100)
        self.ds_3d = xr.Dataset(
            {
                "foo": (["time", "y", "x"], np.random.rand(*shape)),
            },
            {
                "x": (["x"], np.arange(shape[-1])),
                "y": (["y"], np.arange(shape[-2])),
            },
        )

        shape_4d = (10, 50, 100, 3)
        self.ds_4d = xr.Dataset(
            {
                "foo": (["time", "y", "x", "b"], np.random.rand(*shape_4d)),
            },
            {
                "x": (["x"], np.arange(shape_4d[-2])),
                "y": (["y"], np.arange(shape_4d[-3])),
                "b": (["b"], np.arange(shape_4d[-1])),
            },
        )

        self.ds_xy = xr.Dataset(
            {
                "x": (
                    ["sample", "feature"],
                    np.random.random((shape[-1], shape[0])),
                ),
                "y": (["sample"], np.random.random(shape[-1])),
            },
        )


class Generator(Base):
    @parameterized(["preload_batch"], ([True, False]))
    def time_batch_preload(self, preload_batch):
        """
        Construct a generator on a chunked DataSet with and without preloading
        batches.
        """
        ds_dask = self.ds_xy.chunk({"sample": 2})
        BatchGenerator(ds_dask, input_dims={"sample": 2}, preload_batch=preload_batch)

    @parameterized(
        ["input_dims", "batch_dims", "input_overlap"],
        (
            [{"x": 5}, {"x": 10}, {"x": 5, "y": 5}, {"x": 10, "y": 5}],
            [{}, {"x": 20}, {"x": 30}],
            [{}, {"x": 1}, {"x": 2}],
        ),
    )
    def time_batch_input(self, input_dims, batch_dims, input_overlap):
        """
        Benchmark simple batch generation case.
        """
        BatchGenerator(
            self.ds_3d,
            input_dims=input_dims,
            batch_dims=batch_dims,
            input_overlap=input_overlap,
        )

    @parameterized(
        ["input_dims", "concat_input_dims"],
        ([{"x": 5}, {"x": 10}, {"x": 5, "y": 5}], [True, False]),
    )
    def time_batch_concat(self, input_dims, concat_input_dims):
        """
        Construct a generator on a DataSet with and without concatenating
        chunks specified by ``input_dims`` into the batch dimension.
        """
        BatchGenerator(
            self.ds_3d,
            input_dims=input_dims,
            concat_input_dims=concat_input_dims,
        )

    @parameterized(
        ["input_dims", "batch_dims", "concat_input_dims"],
        (
            [{"x": 5}, {"x": 5, "y": 5}],
            [{}, {"x": 10}, {"x": 10, "y": 10}],
            [True, False],
        ),
    )
    def time_batch_concat_4d(self, input_dims, batch_dims, concat_input_dims):
        """
        Construct a generator on a DataSet with and without concatenating
        chunks specified by ``input_dims`` into the batch dimension.
        """
        BatchGenerator(
            self.ds_4d,
            input_dims=input_dims,
            batch_dims=batch_dims,
            concat_input_dims=concat_input_dims,
        )


class Accessor(Base):
    @parameterized(
        ["input_dims"],
        ([{"x": 2}, {"x": 4}, {"x": 2, "y": 2}, {"x": 4, "y": 2}]),
    )
    def time_accessor_input_dim(self, input_dims):
        """
        Benchmark simple batch generation case using xarray accessor
        Equivalent to subset of ``time_batch_input()``.
        """
        self.ds_3d.batch.generator(input_dims=input_dims)


class TorchLoader(Base):
    def setup(self, *args, **kwargs):
        super().setup(**kwargs)
        self.x_gen = BatchGenerator(self.ds_xy["x"], {"sample": 10})
        self.y_gen = BatchGenerator(self.ds_xy["y"], {"sample": 10})

    def time_map_dataset(self):
        """
        Benchmark MapDataset integration with torch DataLoader.
        """
        dataset = MapDataset(self.x_gen, self.y_gen)
        loader = torch.utils.data.DataLoader(dataset)
        iter(loader).next()

    def time_iterable_dataset(self):
        """
        Benchmark IterableDataset integration with torch DataLoader.
        """
        dataset = IterableDataset(self.x_gen, self.y_gen)
        loader = torch.utils.data.DataLoader(dataset)
        iter(loader).next()
