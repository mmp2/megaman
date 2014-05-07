from nose.tools import assert_true
from nose.tools import assert_equal
import scipy.io
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import isspmatrix
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from nose.tools import assert_raises
from nose.plugins.skip import SkipTest

from ..embedding.geometry import *
from ..embedding.spectral_embedding_ import _graph_is_connected

def _load_test_data():
    """ Loads a .mat file from . that contains the following dense matrices
    test_dist_matrix
    Lsymnorm, Lunnorm, Lgeom, Lreno1_5, Lrw
    rad = scalar, radius used in affinity calculations, Laplacians
    """
    xdict = scipy.io.loadmat('Mmani/tests/testMmani_laplacian_rad0_2_lam1_5_n200.mat')
    rad = xdict[ 'rad' ]
    test_dist_matrix = xdict[ 'A' ]
    Lsymnorm = xdict[ 'Lsymnorm' ] 
    Lunnorm = xdict[ 'Lunnorm' ] 
    Lgeom = xdict[ 'Lgeom' ] 
    Lrw = xdict[ 'Lrw' ] 
    Lreno1_5 = xdict[ 'Lreno1_5' ] 
    A = xdict[ 'A' ]

    return rad, test_dist_matrix, A, Lsymnorm, Lunnorm, Lgeom, Lreno1_5, Lrw

def test_laplacian_unknown_normalization():
    """Test that laplacian fails with an unknown normalization type"""
    A = np.array([[ 5, 2, 1 ], [ 2, 3, 2 ],[1,2,5]]) 
    assert_raises(ValueError, graph_laplacian, A, normed='<unknown>')

def test_laplacian_create_A_sparse():
    """
    Test that A_sparse is the same as A_dense for a small A matrix
    """
    rad = 2.
    n_samples = 6
    X = np.arange(n_samples)
    X = X[ :,np.newaxis]
    X = np.concatenate((X,np.zeros((n_samples,1),dtype=float)),axis=1)
    #X = np.asarray( X, order="C" )
    test_dist_matrix = distance_matrix( X, mode = 'radius_neighbors', neighbors_radius = rad )

    A_dense = affinity_matrix( test_dist_matrix.toarray(), rad )
    A_sparse = affinity_matrix( sparse.csr_matrix( test_dist_matrix ), rad )
    A_spdense = A_sparse.toarray()
    A_spdense[ A_spdense == 0 ] = 1.

    print( 'A_dense')
    print( A_dense )
    print( 'A_sparse',  A_spdense )

    assert_array_equal( A_dense, A_spdense )


def test_equal_original():

    rad, test_dist_matrix, Atest, Lsymnorm, Lunnorm, Lgeom, Lreno1_5, Lrw = _load_test_data()

    rad = rad[0]
    rad = rad[0]
    print( type(test_dist_matrix), type( Atest))
    A_dense = affinity_matrix( test_dist_matrix, rad )
    A_sparse = affinity_matrix( sparse.csr_matrix( test_dist_matrix ), rad )
    print( type( A_dense ), type( A_sparse ))
    B = A_sparse.toarray()
    B[ B == 0. ] = 1.
    assert_array_equal( A_dense, B )
#    assert_array_almost_equal( Atest, A_dense, 1 ) fails!!
#    A != Atest

    for (A, issparse) in [(Atest, False), (A_sparse, True)]:
        for (Ltest, normed ) in [(Lsymnorm, 'symmetricnormalized'), (Lunnorm, 'unnormalized'), (Lgeom, 'geometric'), (Lrw, 'randomwalk'), (Lreno1_5, 'renormalized')]:
            L, diag =  graph_laplacian(A, normed=normed, symmetrize=True, scaling_epps=rad, renormalization_exponent=1.5, return_diag=True)
            if issparse:
                print( 'sparse ', normed )
                assert_array_almost_equal( L.todense(), Ltest, 5 )
                assert_array_equal(diag, triu(tril(L)).data)
            else:
                print( 'dense ', normed )
                assert_array_almost_equal( L, Ltest, 5 )
                di = np.diag_indices( L.shape[0] )
                assert_array_equal(diag, np.array( L[di] )) 
            
#TODO: test symmetry 
# test readius
