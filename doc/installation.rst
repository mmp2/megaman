Installation
============

Though ``megaman`` has a fair number of compiled dependencies, it is
straightforward to install using the cross-platform conda_ package manager.

Installation with Conda
-----------------------

To install ``megaman`` and all its dependencies using conda_, run::

    $ conda install megaman --channel=conda-forge

Currently builds are available for OSX and Linux, on Python 2.7, 3.4, and 3.5.
For other operating systems, see the full install instructions below.

Installation from Source
------------------------

To install ``megaman`` from source requires the following:

- python_: tested with versions 2.7, 3.4, and 3.5
- numpy_: version 1.8 or higher
- scipy_: version 0.16.0 or higher
- scikit-learn_: version 0.16.0 or higher
- FLANN_: version 1.8 or higher
- cython_: version 0.23 or higher
- a C++ compiler such as ``gcc``/``g++`` (we recommend version 4.8.*)

Optional requirements include:

- pyamg_, which provides fast decompositions of large sparse matrices
- pyflann_, which offers an alternative FLANN interface for computing distance matrices (this is bundled with the FLANN source code)
- nose_ for running the unit tests

These requirements can be installed on Linux and MacOSX using the following conda command::

    $ conda install --channel=jakevdp pip nose coverage gcc cython numpy scipy scikit-learn pyflann pyamg

Finally, within the source repository, run this command to install the ``megaman`` package itself::

    $ python setup.py install

Unit Tests
----------
``megaman`` uses nose_ for unit tests. To run the unit tests once ``nose`` is installed, type in the source directory::

    $ make test

or, outside the source directory once ``megaman`` is installed::

    $ nosetests megaman

``megaman`` is tested on Python versions 2.7, 3.4, and 3.5.

.. _conda: http://conda.pydata.org/miniconda.html
.. _python: http://python.org
.. _numpy: http://numpy.org
.. _scipy: http://scipy.org
.. _scikit-learn: http://scikit-learn.org
.. _FLANN: http://www.cs.ubc.ca/research/flann/
.. _pyamg: http://pyamg.org/
.. _pyflann: http://www.cs.ubc.ca/research/flann/
.. _nose: https://nose.readthedocs.org/
.. _cython: http://cython.org/
