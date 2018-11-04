xbatcher: Batch Generation from Xarray Datasets
===============================================

Xbatcher is a small library for iterating xarray DataArrays in batches. The
goal is to make it easy to feed xarray datasets to machine learning libraries
such as Keras_.

.. _Keras: https://keras.io/

Installation
------------

No package yet. Install from git:

    pip install git+https://github.com/rabernat/xbatcher.git


Basic Usage
-----------

Let's say we have an xarray dataset

.. ipython:: python

    import xarray as xr
    import numpy as np
    da = xr.DataArray(np.random.rand(1000, 100, 100), name='foo',
                      dims=['time', 'y', 'x']).chunk({'time': 1})
    da

and we want to create batches along the time dimension. We can do it like this

.. ipython:: python

    import xbatcher
    bgen = xbatcher.BatchGenerator(da, {'time': 10})
    for batch in bgen:
        pass
        # actually feed to machine learning library
    batch

API
---

.. autoclass:: xbatcher.BatchGenerator
   :members:
