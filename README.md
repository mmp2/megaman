# Mmani: Scalable manifold learning

Mmani is a scalable manifold learning package implemented in 
python. It has a front-end API designed to be familiar
to [scikit-learn](http://scikit-learn.org/) but harnesses
the C++ Fast Library for Approximate Nearest Neighbors (FLANN)
and the Sparse Symmetric Positive Definite (SSPD) solver
Locally Optimal Block Precodition Gradient (LOBPCG) method
to scale manifold learning algorithms to large data sets.
On a personal computer Mmani can embed 1 million data points
with hundreds of dimensions in 10 minutes. 
Mmani is designed for researchers and as such caches intermediary
steps and indices to allow for fast re-computation with new parameters. 

## Examples

To Do: turn the example.py into an ipython notebook

## Installation
To Do: Work in conda & pip for installing Mmani

This package can be run as pure python with the following required
dependencies:

- [numpy](http://numpy.org) version 1.8 or higher
- [scipy](http://scipy.org) version 0.16.0 or higher

For optimal performance we also require:

- [cython](http://cython.org/)
- [pyamg](http://pyamg.org/)
- [FLANN](http://www.cs.ubc.ca/research/flann/)

## Unit Tests
Mmani uses ``nose`` for unit tests. With nosetests installed, type

    $ nosetests Mmani

to run the unit tests.

The tests are run on Python versions 2.7

To Do: test more python versions

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
