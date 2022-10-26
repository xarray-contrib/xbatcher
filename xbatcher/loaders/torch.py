from typing import Any, Callable, Optional, Tuple

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "The Xbatcher PyTorch Dataset API depends on PyTorch. Please "
        "install PyTorch to proceed."
    ) from exc

# Notes:
# This module includes two PyTorch datasets.
#  - The MapDataset provides an indexable interface
#  - The IterableDataset provides a simple iterable interface
# Both can be provided as arguments to the the Torch DataLoader
# Assumptions made:
#  - Each dataset takes pre-configured X/y xbatcher generators (may not always want two generators in a dataset)
# TODOs:
#  - sort out xarray -> numpy pattern. Currently there is a hardcoded variable name for x/y
#  - need to test with additional dataset parameters (e.g. transforms)


class MapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X_generator,
        y_generator,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        PyTorch Dataset adapter for Xbatcher

        Parameters
        ----------
        X_generator : xbatcher.BatchGenerator
        y_generator : xbatcher.BatchGenerator
        transform : callable, optional
            A function/transform that takes in an array and returns a transformed version.
        target_transform : callable, optional
            A function/transform that takes in the target and transforms it.
        """
        self.X_generator = X_generator
        self.y_generator = y_generator
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.X_generator)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if len(idx) == 1:
                idx = idx[0]
            else:
                raise NotImplementedError(
                    f"{type(self).__name__}.__getitem__ currently requires a single integer key"
                )

        # TODO: figure out the dataset -> array workflow
        # currently hardcoding a variable name
        X_batch = self.X_generator[idx].torch.to_tensor()
        y_batch = self.y_generator[idx].torch.to_tensor()

        if self.transform:
            X_batch = self.transform(X_batch)

        if self.target_transform:
            y_batch = self.target_transform(y_batch)
        return X_batch, y_batch


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
