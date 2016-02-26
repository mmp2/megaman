# megaman: Scalable manifold learning

[![build status](http://img.shields.io/travis/mmp2/megaman/master.svg?style=flat)](https://travis-ci.org/mmp2/megaman)
[![license](http://img.shields.io/badge/license-BSD-blue.svg?style=flat)](https://github.com/mmp2/megaman/blob/master/LICENSE)

``megaman`` is a scalable manifold learning package implemented in
python. It has a front-end API designed to be familiar
to [scikit-learn](http://scikit-learn.org/) but harnesses
the C++ Fast Library for Approximate Nearest Neighbors (FLANN)
and the Sparse Symmetric Positive Definite (SSPD) solver
Locally Optimal Block Precodition Gradient (LOBPCG) method
to scale manifold learning algorithms to large data sets.
On a personal computer megaman can embed 1 million data points
with hundreds of dimensions in 10 minutes.
megaman is designed for researchers and as such caches intermediary
steps and indices to allow for fast re-computation with new parameters.

Documentation can be found at http://mmp2.github.io/megaman/

## Examples

See documentation and example.py for usage.

## Installation

megaman has the following dependencies:

- [python](http://python.org) version 2.7
- [numpy](http://numpy.org) version 1.8 or higher
- [scipy](http://scipy.org) version 0.16.0 or higher
- [scikit-learn](http://scikit-learn.org)
- [FLANN](http://www.cs.ubc.ca/research/flann/)

Additionally, installation from source requires [cython](http://cython.org/) and a C++ compiler such as ``gcc``/``g++``

Optional requirements include

- [pyamg](http://pyamg.org/), which allows for faster decompositions of large matrices
- [pyflann](https://github.com/primetang/pyflann) which offers another method of computing distance matrices

The package uses ``nose`` for unit tests

### Installing from source pip and conda

The above requirements can be installed on Linux and Mac OSX using [conda](http://conda.pydata.org/miniconda.html):

```
$ conda install -c https://conda.anaconda.org/jakevdp pip nose coverage cython numpy scipy scikit-learn flann pyamg gcc
$ pip install pyflann
```

Finally, install the ``megaman`` package itself:
```
$ python setup.py install
```

## Unit Tests
megaman uses ``nose`` for unit tests. With nosetests installed, type

    $ make test

to run the unit tests. The tests are run on Python versions 2.7

## Authors
- [James McQueen](http://www.stat.washington.edu/people/jmcq/)
- [Marina Meila](http://www.stat.washington.edu/mmp/)
- [Zhongyue Zhang](https://github.com/Jerryzcn)
- [Jake VanderPlas](http://www.vanderplas.com)

## Future Work

We have the following planned updates for upcoming releases:

- Native support for K-Nearest Neighbors distance (in progress)
- Lazy R-metric (only calcualte on selected points)
- Make cover_plotter.py work more generally with rmetric.py
