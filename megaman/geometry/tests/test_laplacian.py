from __future__ import division ## removes integer division

import os
import numpy as np
from scipy import io
from scipy import sparse
from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial.distance import pdist, squareform
from megaman.geometry.distance import distance_matrix
import megaman.geometry.geometry as geom
from megaman.geometry.distance import distance_matrix
from megaman.geometry.geometry import laplacian_types
from megaman.geometry.affinity import compute_affinity_matrix
from megaman.geometry.laplacian import compute_laplacian_matrix

random_state = np.random.RandomState(36)
n_sample = 10
d = 2
X = random_state.randn(n_sample, d)
D = squareform(pdist(X))
D[D > 1/d] = 0

TEST_DATA = os.path.join(os.path.dirname(__file__),
                        'testmegaman_laplacian_rad0_2_lam1_5_n200.mat')

def _load_test_data():
    """ Loads a .mat file from . that contains the following dense matrices
    test_dist_matrix
    Lsymnorm, Lunnorm, Lgeom, Lreno1_5, Lrw
    rad = scalar, radius used in affinity calculations, Laplacians
        Note: rad is returned as an array of dimension 1. Outside one must
        make it a scalar by rad = rad[0]
    """
    xdict = io.loadmat(TEST_DATA)

    rad = xdict[ 'rad' ]
    test_dist_matrix = xdict[ 'S' ] # S contains squared distances
    test_dist_matrix = np.sqrt( test_dist_matrix )
    Lsymnorm = xdict[ 'Lsymnorm' ]
    Lunnorm = xdict[ 'Lunnorm' ]
    Lgeom = xdict[ 'Lgeom' ]
    Lrw = xdict[ 'Lrw' ]
    Lreno1_5 = xdict[ 'Lreno1_5' ]
    A = xdict[ 'A' ]
    return rad, test_dist_matrix, A, Lsymnorm, Lunnorm, Lgeom, Lreno1_5, Lrw

def test_laplacian_unknown_method():
    """Test that laplacian fails with an unknown method type"""
    A = np.array([[ 5, 2, 1 ], [ 2, 3, 2 ],[1,2,5]])
    assert_raises(ValueError, compute_laplacian_matrix, A, method='<unknown>')

def test_equal_original(almost_equal_decimals = 5):
    """ Loads the results from a matlab run and checks that our results
    are the same. The results loaded are A the similarity matrix and
    all the Laplacians, sparse and dense.
    """

    rad, test_dist_matrix, Atest, Lsymnorm, Lunnorm, Lgeom, Lreno1_5, Lrw = _load_test_data()

    rad = rad[0]
    rad = rad[0]
    A_dense = compute_affinity_matrix(test_dist_matrix, method = 'auto', 
                                      radius = rad)
    A_sparse = compute_affinity_matrix(sparse.csr_matrix(test_dist_matrix), 
                                       method = 'auto', radius=rad)
    B = A_sparse.toarray()
    B[ B == 0. ] = 1.
    assert_array_almost_equal( A_dense, B, almost_equal_decimals )
    assert_array_almost_equal( Atest, A_dense, almost_equal_decimals )
    for (A, issparse) in [(Atest, False), (sparse.coo_matrix(Atest), True)]:
        for (Ltest, method ) in [(Lsymnorm, 'symmetricnormalized'), 
                                 (Lunnorm, 'unnormalized'), (Lgeom, 'geometric'), 
                                 (Lrw, 'randomwalk'), (Lreno1_5, 'renormalized')]:
            L, diag =  compute_laplacian_matrix(A, method=method, 
                                        symmetrize=True, scaling_epps=rad, 
                                        renormalization_exponent=1.5, 
                                        return_diag=True)
            if issparse:
                assert_array_almost_equal( L.toarray(), Ltest, 5 )
                diag_mask = (L.row == L.col )
                assert_array_equal(diag, L.data[diag_mask].squeeze())
            else:
                assert_array_almost_equal( L, Ltest, 5 )
                di = np.diag_indices( L.shape[0] )
                assert_array_equal(diag, np.array( L[di] ))
