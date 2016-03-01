from __future__ import division ## removes integer division
import numpy as np
from scipy import sparse
import sys, os
from nose import SkipTest
from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from scipy.spatial.distance import pdist, squareform
import warnings
from megaman.geometry.distance import distance_matrix
import time

random_state = np.random.RandomState(36)
n_sample = 10
d = 2
X = random_state.randn(n_sample, d)
D = squareform(pdist(X))
D[D > 1/d] = 0

def test_all_methods_close():
    D = distance_matrix(X)

    def check_method(method):
        if method == 'pyflann':
            try:
                import pyflann as pyf
            except ImportError:
                raise SkipTest("pyflann not installed.")
            flindex = pyf.FLANN()
            flparams = flindex.build_index(X, algorithm='kmeans',
                                           target_precision=0.9)
        else:
            flindex = None
        this_D = distance_matrix(X, method=method, flindex=flindex)
        assert_allclose(this_D.toarray(), D.toarray(), rtol=1E-5)

    for method in ['cyflann', 'pyflann', 'brute']:
        yield check_method, method


def test_flindex_passed():
    assert_raises(ValueError, distance_matrix, X, 'pyflann')


def test_unknown_method():
    assert_raises(ValueError, distance_matrix, X, 'foo')
