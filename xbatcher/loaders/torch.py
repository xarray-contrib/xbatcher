from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

import xarray as xr

from xbatcher import BatchGenerator

try:
    import torch
except ImportError as exc:
    raise ImportError(
        'The Xbatcher PyTorch Dataset API depends on PyTorch. Please '
        'install PyTorch to proceed.'
    ) from exc

try:
    import dask
except ImportError:
    dask: ModuleType | None = None  # type: ignore[no-redef]

T_DataArrayOrSet = xr.DataArray | xr.Dataset

# Notes:
# This module includes two PyTorch datasets.
#  - The MapDataset provides an indexable interface
#  - The IterableDataset provides a simple iterable interface
# Both can be provided as arguments to the the Torch DataLoader
# Assumptions made:
#  - Each dataset takes pre-configured X/y xbatcher generators (may not always want two generators in a dataset)
# TODOs:
#  - need to test with additional dataset parameters (e.g. transforms)


def to_tensor(xr_obj: T_DataArrayOrSet) -> torch.Tensor:
    """Convert this DataArray or Dataset to a torch.Tensor"""
    if isinstance(xr_obj, xr.Dataset):
        xr_obj = xr_obj.to_array().squeeze(dim='variable')
    if isinstance(xr_obj, xr.DataArray):
        xr_obj = xr_obj.data
    return torch.tensor(xr_obj)


class MapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X_generator: BatchGenerator,
        y_generator: BatchGenerator | None = None,
        transform: Callable[[T_DataArrayOrSet], torch.Tensor] = to_tensor,
        target_transform: Callable[[T_DataArrayOrSet], torch.Tensor] = to_tensor,
    ) -> None:
        """
        PyTorch Dataset adapter for Xbatcher

        Parameters
        ----------
        X_generator : xbatcher.BatchGenerator
        y_generator : xbatcher.BatchGenerator
        transform, target_transform : callable, optional
            A function/transform that takes in an Xarray object and returns a transformed version in the form of a torch.Tensor.
        """
        self.X_generator = X_generator
        self.y_generator = y_generator
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.X_generator)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if len(idx) == 1:
                idx = idx[0]
            else:
                raise NotImplementedError(
                    f'{type(self).__name__}.__getitem__ currently requires a single integer key'
                )

        # generate batch (or batches)
        if self.y_generator is not None:
            X_batch, y_batch = self.X_generator[idx], self.y_generator[idx]
        else:
            X_batch, y_batch = self.X_generator[idx], None

        # load batch (or batches) with dask if possible
        if dask is not None:
            X_batch, y_batch = dask.compute(X_batch, y_batch)

        # apply transformation(s)
        X_batch_tensor = self.transform(X_batch)
        if y_batch is not None:
            y_batch_tensor = self.target_transform(y_batch)

        assert isinstance(X_batch_tensor, torch.Tensor), self.transform

        if y_batch is None:
            return X_batch_tensor
        assert isinstance(y_batch_tensor, torch.Tensor)
        return X_batch_tensor, y_batch_tensor


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        X_generator,
        y_generator,
    ) -> None:
        """
        PyTorch Dataset adapter for Xbatcher

        Parameters
        ----------
        X_generator : xbatcher.BatchGenerator
        y_generator : xbatcher.BatchGenerator
        """

        self.X_generator = X_generator
        self.y_generator = y_generator

    def __iter__(self):
        for xb, yb in zip(self.X_generator, self.y_generator):
            yield (xb.torch.to_tensor(), yb.torch.to_tensor())
