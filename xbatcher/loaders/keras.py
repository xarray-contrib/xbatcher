from collections.abc import Callable
from typing import Any

try:
    import tensorflow as tf
except ImportError as exc:
    raise ImportError(
        'The Xbatcher TensorFlow Dataset API depends on TensorFlow. Please '
        'install TensorFlow to proceed.'
    ) from exc

# Notes:
# This module includes one Keras dataset, which can be provided to model.fit().
#  - The CustomTFDataset provides an indexable interface
# Assumptions made:
#  - The dataset takes pre-configured X/y xbatcher generators (may not always want two generators in a dataset)


class CustomTFDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        X_generator,
        y_generator,
        *,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """
        Keras Dataset adapter for Xbatcher

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

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        X_batch = tf.convert_to_tensor(self.X_generator[idx].data)
        y_batch = tf.convert_to_tensor(self.y_generator[idx].data)

        # TODO: Should the transformations be applied before tensor conversion?
        if self.transform:
            X_batch = self.transform(X_batch)

        if self.target_transform:
            y_batch = self.target_transform(y_batch)
        return X_batch, y_batch
