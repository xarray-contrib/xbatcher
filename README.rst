xbatcher: Batch Generation from Xarray Datasets
===============================================

|Build Status| |codecov| |docs| |pypi| |conda-forge| |license|


Xbatcher is a small library for iterating Xarray DataArrays and Datasets in
batches. The goal is to make it easy to feed Xarray objects to machine
learning libraries such as PyTorch_ or TensorFlow_. View the |docs| for more
info.

.. _TensorFlow: https://www.tensorflow.org/

.. _PyTorch: https://pytorch.org/


.. |Build Status| image:: https://github.com/xarray-contrib/xbatcher/workflows/CI/badge.svg
   :target: https://github.com/xarray-contrib/xbatcher/actions
   :alt: github actions build status
.. |codecov| image:: https://codecov.io/gh/xarray-contrib/xbatcher/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/xarray-contrib/xbatcher
   :alt: code coverage
.. |docs| image:: http://readthedocs.org/projects/xbatcher/badge/?version=latest
   :target: http://xbatcher.readthedocs.org/en/latest/?badge=latest
   :alt: docs
.. |pypi| image:: https://img.shields.io/pypi/v/xbatcher.svg
   :target: https://pypi.python.org/pypi/xbatcher
   :alt: pypi
.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/xbatcher.svg
   :target: https://anaconda.org/conda-forge/xbatcher
   :alt: conda-forge
.. |license| image:: https://img.shields.io/github/license/xarray-contrib/xbatcher.svg
   :target: https://github.com/xarray-contrib/xbatcher
   :alt: license

Installation
------------

Xbatcher can be installed from PyPI as::

    python -m pip install xbatcher

Or via Conda as::

    conda install -c conda-forge xbatcher

Or from source as::

    python -m pip install git+https://github.com/xarray-contrib/xbatcher.git

.. note::
   The required dependencies installed with Xbatcher are `Xarray <https://xarray.dev/>`_,
   `Dask <https://www.dask.org/>`_, and `NumPy <https://numpy.org/>`_.
   You will need to separately install `TensorFlow <https://www.tensorflow.org/>`_
   or `PyTorch <https://pytorch.org/>`_ to use those data loaders or
   Xarray accessors. `Review the installation instructions <https://xbatcher.readthedocs.io/en/latest/#optional-dependencies>`_
   for more details.

Documentation
-------------

Documentation is hosted on ReadTheDocs: https://xbatcher.readthedocs.org

License
------------

Apache License 2.0, see LICENSE file.

Acknowledgements
----------------

This work was funded in part by:

NASA ACCESS19-0049: Pangeo ML: Open Source Tools and Pipelines for Scalable Machine Learning Using NASA Earth Observation Data

This work was motivated by many conversations in the Pangeo community and Pangeo ML working group
