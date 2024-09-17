xbatcher: Batch Generation from Xarray Datasets
===============================================

Xbatcher is a small library for iterating Xarray DataArrays and Datasets in
batches. The goal is to make it easy to feed Xarray objects to machine learning
libraries such as Keras_.

.. _Keras: https://keras.io/

Installation
------------

Xbatcher can be installed from PyPI as::

    python -m pip install xbatcher

Or via Conda as::

    conda install -c conda-forge xbatcher

Or from source as::

    python -m pip install git+https://github.com/xarray-contrib/xbatcher.git

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

.. note::
    The required dependencies installed with Xbatcher are `Xarray <https://xarray.dev/>`_,
    `Dask <https://www.dask.org/>`_, and `NumPy <https://numpy.org/>`_.
    You will need to separately install `TensorFlow <https://www.tensorflow.org/>`_
    or `PyTorch <https://pytorch.org/>`_ to use those data loaders or
    Xarray accessors.

To install Xbatcher and PyTorch via `Conda <https://docs.conda.io/>`_::

    conda install -c conda-forge xbatcher pytorch

Or via PyPI::

    python -m pip install xbatcher[torch]

To install Xbatcher and TensorFlow via `Conda <https://docs.conda.io/>`_::

    conda install -c conda-forge xbatcher tensorflow

Or via PyPI::

    python -m pip install xbatcher[tensorflow]

Basic Usage
-----------

Let's say we have an Xarray Dataset

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

or via a built-in `Xarray accessor <http://xarray.pydata.org/en/stable/internals/extending-xarray.html#extending-xarray>`_:

.. ipython:: python

    import xbatcher

    for batch in da.batch.generator({'time': 10}):
        pass
        # actually feed to machine learning library
    batch

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   user-guide/index
   tutorials-and-presentations
   roadmap
   contributing
