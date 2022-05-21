.. _api:

API reference
-------------

This page provides an auto-generated summary of Xbatcher's API.

Dataset.batch and DataArray.batch
=================================

.. currentmodule:: xarray

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Dataset.batch.generator
   DataArray.batch.generator

Core
====

.. autoclass:: xbatcher.BatchGenerator
   :members:

Dataloaders
===========
.. autoclass:: xbatcher.loaders.torch.MapDataset
   :members:

.. autoclass:: xbatcher.loaders.torch.IterableDataset
   :members:

.. autoclass:: xbatcher.loaders.keras.CustomTFDataset
   :members:
