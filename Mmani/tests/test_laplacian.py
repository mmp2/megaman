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


def test_equal_original():

    rad, test_dist_matrix, A, Lsymnorm, Lunnorm, Lgeom, Lreno1_5, Lrw = _load_test_data()

    A_dense = affinity_matrix( test_dist_matrix, rad )
    A_sparse = affinity_matrix( sparse.csr_matrix( test_dist_matrix ), rad )
    print A_sparse.data.shape
    B = A_sparse.todense()
    assert_array_equal( A_dense, A_sparse.todense() )

    for (A, issparse) in [(A_dense, False), (A_sparse, True)]:
        for (Ltest, normed ) in [(Lsymnorm, 'symmetricnormailized'), (Lunnorm, 'unnormalized'), (Lgeom, 'geometric'), (Lrw, 'randomwalk'), (Lreno1_5, 'renormalized')]:
            L, diag =  graph_laplacian(A, normed=normed, symmetrize=True, scaling_epps=0., renormalization_exponent=1.5, return_diag=true)
            if issparse:
                assert_array_almost_equal( L.todense(), Ltest, 5 )
                assert_array_equal(diag, triu(tril(L)).data)
            else:
                assert_array_almost_equal( L, Ltest, 5 )
                di = np.diag.indices( L.shape )
                assert_array_equal(diag, np.array( L.di )) 
            
#TODO: test symmetry 
# test readius
