.. _roadmap:

Roadmap
=======

Authors: Joe Hamman and Ryan Abernathey
Date: February 7, 2019

Background and scope
--------------------

Xbatcher is a small library for iterating xarray objects in batches. The
goal is to make it easy to feed xarray datasets to machine learning libraries
such as `Keras`_ or `PyTorch`_. For example, implementing a simple machine
learning workflow may look something like this:

.. code-block:: Python

    import xarray as xr
    import xbatcher as xb

    da = xr.open_dataset(filename, chunks=chunks)     # open a dataset and use dask
    da_train = preprocess(ds)                         # perform some preprocessing
    bgen = xb.BatchGenerator(da_train, {'time': 10})  # create a generator

    for batch in bgen:                                # iterate through the generator
        model.fit(batch['x'], batch['y'])             # fit a deep-learning model
        # or
        model.predict(batch['x'])                     # make one batch of predictions

We are currently envisioning the project growing to support more complex
extract-transform-load components commonly found in machine learning workflows
that use multidimensional data. We note that many of the concepts in Xbatcher
have been developed through collaborations in the `Pangeo Project Machine
Learning Working Group <https://pangeo.io/meeting-notes.html>`_.

Batch generation
~~~~~~~~~~~~~~~~

At the core of Xbatcher is the ability to define a schema that defines a
selection of a larger dataset. Today, this schema is fairly simple (e.g.
`{'time': 10}`) but this may evolve in the future. As we describe below,
additional utilities for shuffling, sampling, and caching may provide enhanced
batch generation functionality

Shuffle and Sampling APIs
~~~~~~~~~~~~~~~~~~~~~~~~~

When training machine-learning models in batches, it is often necessary to
selectively or randomly sample from your training data. Xbatcher can help
facilitate seamless shuffling and sampling by providing APIs that operate on
batches and/or full datasets. This may require working with Xarray and Dask to
facilitate fast, distributed shuffles of Dask arrays.

Caching APIs
~~~~~~~~~~~~

A common pattern in ML is perform the ETL tasks once before saving the results
to a local file system. This is an effective approach for speeding up dataset
loading during training but comes with numerous downsides (i.e. requires
sufficient file space, breaks workflow continuity, etc.). We propose the
development of a pluggable cache mechanism in Xbatcher that would help address
these downsides while providing improved performance during model training and
inference. For example, this pluggable cache mechanism may allow choosing
between multiple cache types, such as an LRU in-memory cache, a Zarr filesystem
or S3 bucket, or a Redis database cache.

Integration with TensorFlow and PyTorch Dataset Loaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deep-learning libraries like TensorFlow and PyTorch provide high-performance
dataset-generator APIs that facilitate the construction of flexible and
efficient input pipelines. In particular, they have been optimized to support
asynchronous data loading and training, transfer to and from GPUs, and batch
caching. Xbatcher will provide compatible dataset APIs that allow users to pass
Xarray datasets directly to deep-learning frameworks.

Dependencies
------------

- Core: Xarray, Pandas, Dask, Scikit-learn, Numpy, Scipy
- Optional: Keras, PyTorch, Tensorflow, etc.

.. _Keras: https://keras.io/
.. _PyTorch: https://pytorch.org/
