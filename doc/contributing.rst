.. _contributing:

************************
Contributing to xbatcher
************************

.. note::

  Large parts of this document came from the `Xarray Contributing
  Guide <http://docs.xarray.dev/en/stable/contributing.html>`_, which is based
  on the `Pandas Contributing Guide
  <http://pandas.pydata.org/pandas-docs/stable/contributing.html>`_.

Running the performance test suite
----------------------------------

*xbatcher* is starting a suite of benchmarking tests using
`asv <https://github.com/airspeed-velocity/asv>`__ to enable easy monitoring of
the performance of critical operations. These benchmarks are all found in the
``asv_bench`` directory.

To use all features of asv, you will need either ``conda`` or ``virtualenv``.
For more details please check the `asv installation webpage
<https://asv.readthedocs.io/en/latest/installing.html>`_.

To install asv::

    pip install git+https://github.com/airspeed-velocity/asv

If you need to run a benchmark, change your directory to ``asv_bench/`` and run::

    asv continuous -f 1.1 main <my-branch>

You can replace ``my-branch`` with the name of the branch you are working on.
The output will include "BENCHMARKS NOT SIGNIFICANTLY CHANGED" if the
benchmarks did not change by more than 10%.

The command uses ``conda`` by default for creating the benchmark
environments. If you want to use virtualenv instead, write::

    asv continuous -f 1.1 -E virtualenv main <my-branch>

The ``-E virtualenv`` option should be added to all ``asv`` commands
that run benchmarks. The default value is defined in ``asv.conf.json``.

If you want to only run a specific group of tests from a file, you can do it
using ``.`` as a separator. For example::

    asv continuous -f 1.1 main HEAD -b benchmarks.Generator.time_batch_preload

will only run the ``Generator.time_batch_preload`` benchmark defined in
``benchmarks.py``.

Information on how to write a benchmark and how to use asv can be found in the
`asv documentation <https://asv.readthedocs.io/en/latest/writing_benchmarks.html>`_.
