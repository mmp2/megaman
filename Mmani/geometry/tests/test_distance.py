from __future__ import division ## removes integer division
import numpy as np
from scipy import sparse
import sys
from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial.distance import pdist, squareform
import warnings
from Mmani.geometry.distance import distance_matrix

random_state = np.random.RandomState(36)
n_sample = 10
d = 2
X = random_state.randn(n_sample, d)
D = squareform(pdist(X)) 
D[D > 1/d] = 0

def test_all_methods_close(almost_equal_decimals = 5):
    try:
        import pyflann as pyf
        flindex = pyf.FLANN()
        flparams = flindex.build_index(X, algorithm = 'kmeans', target_precision = 0.9)
        D2 = distance_matrix(X, method = 'pyflann', flindex = flindex) 
        assert_array_almost_equal(D2.todense(), D, almost_equal_decimals)
    except ImportError:
        try: 
            # This path is only valid on newton.stat.washington.edu 
            path_to_flann = '/homes/jmcq/flann-1.8.4-src/src/python'
            sys.path.insert(0, path_to_flann)
            import pyflann as pyf
            flindex = pyf.FLANN()
            flparams = flindex.build_index(X, algorithm = 'kmeans', target_precision = 0.9)
            D2 = distance_matrix(X, method = 'pyflann', flindex = flindex) 
            assert_array_almost_equal(D2.todense(), D, almost_equal_decimals)
        except ImportError:
            warnings.warn("pyflann not installed. Will not test pyflann method")
    
    D1 = distance_matrix(X, method = 'auto') 
    D3 = distance_matrix(X, method = 'cyflann')
    D4 = distance_matrix(X, method = 'brute')
    D5 = distance_matrix(X)

    assert_array_almost_equal(D1.todense(), D, almost_equal_decimals)
    assert_array_almost_equal(D3.todense(), D, almost_equal_decimals)
    assert_array_almost_equal(D4.todense(), D, almost_equal_decimals)
    assert_array_almost_equal(D5.todense(), D, almost_equal_decimals)
    
def test_flindex_passed():
    assert_raises(ValueError, distance_matrix, X, 'pyflann')
    return True

def test_unknown_method():
    assert_raises(ValueError, distance_matrix, X, 'foo')
    return True