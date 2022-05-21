from typing import Any, Callable, Optional, Tuple

import tensorflow as tf
import xarray as xr

# Notes:
# This module includes one Keras dataset, which can be provided to model.fit().
#  - The CustomTFDataset provides an indexable interface
# Assumptions made:
#  - The dataset takes pre-configured X/y xbatcher generators (may not always want two generators in a dataset)
# TODOs:
#  - need to test with additional dataset parameters (e.g. transforms)


class CustomTFDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        X_generator,
        y_generator,
        *,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dim: str = 'new_dim',
    ) -> None:
        '''
        Keras Dataset adapter for Xbatcher

        Parameters
        ----------
        X_generator : xbatcher.BatchGenerator
        y_generator : xbatcher.BatchGenerator
        transform : callable, optional
            A function/transform that takes in an array and returns a transformed version.
        target_transform : callable, optional
            A function/transform that takes in the target and transforms it.
        dim : str, 'new_dim'
            Name of dim to pass to :func:`xarray.concat` as the dimension
            to concatenate all variables along.
        '''
        self.X_generator = X_generator
        self.y_generator = y_generator
        self.transform = transform
        self.target_transform = target_transform
        self.concat_dim = dim

    def __len__(self) -> int:
        return len(self.X_generator)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        X_batch = tf.convert_to_tensor(
            xr.concat(
                (
                    self.X_generator[idx][key]
                    for key in list(self.X_generator[idx].keys())
                ),
                self.concat_dim,
            ).data
        )
        y_batch = tf.convert_to_tensor(
            xr.concat(
                (
                    self.y_generator[idx][key]
                    for key in list(self.y_generator[idx].keys())
                ),
                self.concat_dim,
            ).data
        )

        # TODO: Should the transformations be applied before tensor conversion?
        if self.transform:
            X_batch = self.transform(X_batch)

        if self.target_transform:
            y_batch = self.target_transform(y_batch)
        return X_batch, y_batch
