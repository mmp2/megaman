from __future__ import division ## removes integer division

import os
import numpy as np
from scipy import sparse
from numpy.testing import assert_array_equal
from scipy.spatial.distance import pdist, squareform
from megaman.geometry.distance import compute_adjacency_matrix
from megaman.geometry.affinity import compute_affinity_matrix

random_state = np.random.RandomState(36)
n_sample = 10
d = 2
X = random_state.randn(n_sample, d)
D = squareform(pdist(X))
D[D > 1/d] = 0

TEST_DATA = os.path.join(os.path.dirname(__file__),
                        'testmegaman_laplacian_rad0_2_lam1_5_n200.mat')

def test_affinity_sparse_vs_dense():
    """
    Test that A_sparse is the same as A_dense for a small A matrix
    """
    rad = 2.
    n_samples = 6
    X = np.arange(n_samples)
    X = X[ :,np.newaxis]
    X = np.concatenate((X,np.zeros((n_samples,1),dtype=float)),axis=1)
    X = np.asarray( X, order="C" )
    test_dist_matrix = compute_adjacency_matrix( X, method = 'auto', radius = rad )
    A_dense = compute_affinity_matrix(test_dist_matrix.toarray(), method = 'auto', 
                                      radius = rad, symmetrize = False )
    A_sparse = compute_affinity_matrix(sparse.csr_matrix(test_dist_matrix), 
                                       method = 'auto', radius = rad, symmetrize = False)
    A_spdense = A_sparse.toarray()
    A_spdense[ A_spdense == 0 ] = 1.
    assert_array_equal( A_dense, A_spdense )