from __future__ import division ## removes integer division

import sys
import warnings
import numpy as np
from scipy import sparse
from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial.distance import pdist, squareform
from Mmani.geometry.geometry import * 
from Mmani.geometry.distance import distance_matrix

random_state = np.random.RandomState(36)
n_sample = 10
d = 2
X = random_state.randn(n_sample, d)
D = squareform(pdist(X)) 
D[D > 1/d] = 0

def test_Geometry_distance(almost_equal_decimals = 5):
    geom = Geometry(X)
    D1 = geom.get_distance_matrix()
    D2 = distance_matrix(X)
    assert_array_almost_equal(D1.todense(), D2.todense(), almost_equal_decimals)

