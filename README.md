# megaman: Manifold Learning for Millions of Points

<img src="https://raw.githubusercontent.com/mmp2/megaman/master/doc/images/word2vec_rmetric_plot_no_digits.png" height=200><img src="https://raw.githubusercontent.com/mmp2/megaman/master/doc/images/spectra_D4000.png" height=200><img src="https://raw.githubusercontent.com/mmp2/megaman/master/doc/images/spectra_Halpha.png" height=200>

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/megaman/badges/downloads.svg)](https://anaconda.org/conda-forge/megaman)
[![build status](http://img.shields.io/travis/mmp2/megaman/master.svg?style=flat)](https://travis-ci.org/mmp2/megaman)
[![version status](http://img.shields.io/pypi/v/megaman.svg?style=flat)](https://pypi.python.org/pypi/megaman)
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

Package documentation can be found at http://mmp2.github.io/megaman/

If you use our software please cite the following JMLR paper:

McQueen, Meila, VanderPlas, & Zhang, "Megaman: Scalable Manifold Learning in Python",
Journal of Machine Learning Research, Vol 17 no. 14, 2016.
http://jmlr.org/papers/v17/16-109.html

You can also find our arXiv paper at http://arxiv.org/abs/1603.02763

## Examples

- [Tutorial Notebook]( https://github.com/mmp2/megaman/blob/master/examples/megaman_tutorial.ipynb)

## Installation with Conda

The easiest way to install ``megaman`` and its dependencies is with
[conda](http://conda.pydata.org/miniconda.html), the cross-platform package
manager for the scientific Python ecosystem.

To install megaman and its dependencies, run

```
$ conda install megaman --channel=conda-forge
```

Currently builds are available for OSX and Linux, on Python 2.7, 3.4, and 3.5.
For other operating systems, see the full install instructions below.

## Installation from source

To install megaman from source requires the following:

- [python](http://python.org) tested with versions 2.7, 3.4, and 3.5
- [numpy](http://numpy.org) version 1.8 or higher
- [scipy](http://scipy.org) version 0.16.0 or higher
- [scikit-learn](http://scikit-learn.org)
- [FLANN](http://www.cs.ubc.ca/research/flann/)
- [cython](http://cython.org/)
- a C++ compiler such as ``gcc``/``g++``
- [plotly](https://plot.ly) an graphing library for interactive plot

Optional requirements include

- [pyamg](http://pyamg.org/), which allows for faster decompositions of large matrices
- [pyflann](http://www.cs.ubc.ca/research/flann/) which offers another method of computing distance matrices (this is bundled with the FLANN source code)
- [nose](https://nose.readthedocs.org/) for running the unit tests
- [h5py](http://www.h5py.org) for reading testing .mat files
- [slepc4py] (http://slepc.upv.es/), for parallel eigendecomposition.

These requirements can be installed on Linux and MacOSX using the following conda command:

```
$ conda install --channel=conda-forge pip nose coverage gcc cython numpy scipy scikit-learn pyflann pyamg h5py plotly
```

Finally, within the source repository, run this command to install the ``megaman`` package itself:
```
$ python setup.py install
```

## Unit Tests
megaman uses ``nose`` for unit tests. With ``nose`` installed, type
```
$ make test
```
to run the unit tests. ``megaman`` is tested on Python versions 2.7, 3.4, and 3.5.

## Authors
- [James McQueen](http://www.stat.washington.edu/people/jmcq/)
- [Marina Meila](http://www.stat.washington.edu/mmp/)
- [Zhongyue Zhang](https://github.com/Jerryzcn)
- [Jake VanderPlas](http://www.vanderplas.com)
- [Yu-Chia Chen](https://github.com/yuchaz)

## Other Contributors

- Xiao Wang: lazy rmetric, Nystrom Extension

## Future Work

See this issues list for what we have planned for upcoming releases:

[Future Work](https://github.com/mmp2/megaman/issues/47)
