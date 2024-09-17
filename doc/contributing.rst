.. _contributing:

******************
Contributing Guide
******************

.. note::

  Large parts of this document came from the `Xarray Contributing
  Guide <http://docs.xarray.dev/en/stable/contributing.html>`_, which is based
  on the `Pandas Contributing Guide
  <http://pandas.pydata.org/pandas-docs/stable/contributing.html>`_.

Bug reports and feature requests
================================

To report bugs or request new features, head over to the `xbatcher repository
<https://github.com/xarray-contrib/xbatcher/issues>`_.

Contributing code
==================

`GitHub has instructions <https://help.github.com/set-up-git-redirect>`__ for
installing git, setting up your SSH key, and configuring git.  All these steps
need to be completed for you to work between your local repository and GitHub.

.. _contributing.forking:

Forking
-------

You will need your own fork to work on the code. Go to the `xbatcher project
page <https://github.com/xarray-contrib/xbatcher>`_ and hit the ``Fork`` button.
You will need to clone your fork to your machine::

    git clone git@github.com:yourusername/xbatcher.git
    cd xbatcher
    git remote add upstream git@github.com:xarray-contrib/xbatcher.git

This creates the directory ``xbatcher`` and connects your repository to
the upstream (main project) *xbatcher* repository.

.. _contributing.dev_env:

Creating a development environment
----------------------------------

To test out code changes, you'll need to build *xbatcher* from source, which
requires a Python environment. If you're making documentation changes, you can
skip to :ref:`contributing.documentation` but you won't be able to build the
documentation locally before pushing your changes.

.. _contributiong.dev_python:

Creating a Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before starting any development, you'll need to create an isolated xbatcher
development environment:

- Install either `Anaconda <https://www.anaconda.com/download/>`_ or `miniconda
  <https://conda.io/miniconda.html>`_
- Make sure your conda is up to date (``conda update conda``)
- Make sure that you have :ref:`cloned the repository <contributing.forking>`
- ``cd`` to the *xbatcher* source directory

First we'll create and activate the build environment:

.. code-block:: sh

    conda env create --file ci/requirements/environment.yml
    conda activate xbatcher-tests

At this point you should be able to import *xbatcher* from your locally
built version:

.. code-block:: sh

   $ python  # start an interpreter
   >>> import xbatcher
   >>> xbatcher.__version__

This will create the new environment, and not touch any of your existing environments,
nor any existing Python installation.

To view your environments::

      conda info --envs

To return to your base environment::

      conda deactivate

See the full conda docs `here <http://conda.pydata.org/docs>`__.

Setting up pre-commit
~~~~~~~~~~~~~~~~~~~~~

We use `pre-commit <https://pre-commit.com/>`_ to manage code linting and style.
To set up pre-commit after activating your conda environment, run:

.. code-block:: sh

    pre-commit install

Creating a branch
-----------------

You want your ``main`` branch to reflect only production-ready code, so create a
feature branch before making your changes. For example::

    git branch shiny-new-feature
    git checkout shiny-new-feature

The above can be simplified to::

    git checkout -b shiny-new-feature

This changes your working directory to the shiny-new-feature branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to *xbatcher*. You can have many "shiny-new-features"
and switch in between them using the ``git checkout`` command.

To update this branch, you need to retrieve the changes from the ``main`` branch::

    git fetch upstream
    git merge upstream/main

This will combine your commits with the latest *xbatcher* git ``main``.  If this
leads to merge conflicts, you must resolve these before submitting your pull
request.  If you have uncommitted changes, you will need to ``git stash`` them
prior to updating.  This will effectively store your changes, which can be
reapplied after updating.

Running the test suite
----------------------

*xbatcher* uses the `pytest <https://docs.pytest.org/en/latest/contents.html>`_
framework for testing. You can run the test suite using::

    pytest xbatcher



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

Contributing documentation
==========================

We greatly appreciate documentation improvements. The docs are built from the docstrings
in the code and the docs in the ``doc`` directory.

To build the documentation, you will need to requirements listed in ``ci/requirements/doc.yml``.
You can create an environment for building the documentation using::

    conda env create --file ci/requirements/docs.yml
    conda activate xbatcher-docs

You can then build the documentation using::

    cd docs
    make html

Contributing changes
====================

Once you've made changes, you can see them by typing::

    git status

If you have created a new file, it is not being tracked by git. Add it by typing::

    git add path/to/file-to-be-added.py

The following defines how a commit message should be structured:

    * A subject line with `< 72` chars.
    * One blank line.
    * Optionally, a commit message body.

Now you can commit your changes in your local repository::

    git commit -m

When you want your changes to appear publicly on your GitHub page, push your
commits to a branch off your fork::

    git push origin shiny-new-feature

Here ``origin`` is the default name given to your remote repository on GitHub.
You can see the remote repositories::

    git remote -v

If you navigate to your branch on GitHub, you should see a banner to submit a pull
request to the *xbatcher* repository.

.. _contributing.ci:

Continuous integration
======================

Continuous integration is done with `GitHub Actions <https://docs.github.com/en/actions/learn-github-actions>`_.

There are currently 3 workflows configured:

- `main.yaml <https://github.com/xarray-contrib/xbatcher/blob/main/.github/workflows/main.yaml>`_ - Run test suite with pytest.
- `pypi-release.yaml <https://github.com/xarray-contrib/xbatcher/blob/main/.github/workflows/pypi-release.yaml>`_ - Publish
  wheels to TestPyPI and PyPI on a tagged release. The pull request trigger can be uncommented to test a release using Test PyPI.
- `release-drafter.yml <https://github.com/xarray-contrib/xbatcher/blob/main/.github/workflows/release-drafter.yml>`_ - Draft
  release notes based on PR titles and labels.
