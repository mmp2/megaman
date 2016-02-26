from __future__ import division ## removes integer division

import sys
import os
import warnings
import numpy as np
from scipy import io
from scipy import sparse
from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial.distance import pdist, squareform
from megaman.geometry.geometry import *
from megaman.geometry.distance import distance_matrix
import megaman.geometry.geometry as geom
from megaman.utils.testing import assert_raise_message
from megaman.geometry.distance import distance_matrix
from megaman.geometry.geometry import laplacian_types

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

def test_Geometry_distance(almost_equal_decimals = 5):
    geom = Geometry(X)
    D1 = geom.get_distance_matrix()
    D2 = distance_matrix(X)
    assert_array_almost_equal(D1.todense(), D2.todense(), almost_equal_decimals)

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
    X = np.asarray( X, order="C" )
    test_dist_matrix = distance_matrix( X, radius = rad )

    A_dense = affinity_matrix( test_dist_matrix.toarray(), rad, symmetrize = False )
    A_sparse = affinity_matrix( sparse.csr_matrix( test_dist_matrix ), rad, symmetrize = False )
    A_spdense = A_sparse.toarray()
    A_spdense[ A_spdense == 0 ] = 1.

    print( 'A_dense')
    print( A_dense )
    print( 'A_sparse',  A_spdense )

    assert_array_equal( A_dense, A_spdense )

def test_equal_original(almost_equal_decimals = 5):
    """ Loads the results from a matlab run and checks that our results
    are the same. The results loaded are A the similarity matrix and
    all the Laplacians, sparse and dense.
    """

    rad, test_dist_matrix, Atest, Lsymnorm, Lunnorm, Lgeom, Lreno1_5, Lrw = _load_test_data()

    rad = rad[0]
    rad = rad[0]
    A_dense = affinity_matrix( test_dist_matrix, rad )
    A_sparse = affinity_matrix( sparse.csr_matrix( test_dist_matrix ), rad )
    B = A_sparse.toarray()
    B[ B == 0. ] = 1.
    assert_array_almost_equal( A_dense, B, almost_equal_decimals )
    assert_array_almost_equal( Atest, A_dense, almost_equal_decimals )

    for (A, issparse) in [(Atest, False), (sparse.coo_matrix(Atest), True)]:
        for (Ltest, normed ) in [(Lsymnorm, 'symmetricnormalized'), (Lunnorm, 'unnormalized'), (Lgeom, 'geometric'), (Lrw, 'randomwalk'), (Lreno1_5, 'renormalized')]:
            L, diag =  graph_laplacian(A, normed=normed, symmetrize=True, scaling_epps=rad, renormalization_exponent=1.5, return_diag=True)
            if issparse:
                print( 'sparse ', normed )
                assert_array_almost_equal( L.toarray(), Ltest, 5 )
                diag_mask = (L.row == L.col )
                assert_array_equal(diag, L.data[diag_mask].squeeze())
            else:
                print( 'dense ', normed )
                assert_array_almost_equal( L, Ltest, 5 )
                di = np.diag_indices( L.shape[0] )
                assert_array_equal(diag, np.array( L[di] ))

def test_distance_types(almost_equal_decimals = 5):
    X = np.random.uniform(size=(20,2))
    G1 = geom.Geometry(X, input_type = 'data', distance_method = 'brute',
                        neighborhood_radius = 1)
    G2 = geom.Geometry(X, input_type = 'data', distance_method = 'cyflann',
                        neighborhood_radius = 1)
    #G3 = geom.Geometry(X, input_type = 'data', distance_method = 'pyflann',
    #                    neighborhood_radius = 1, path_to_flann = path_to_flann)
    d1 = G1.get_distance_matrix()
    d2 = G2.get_distance_matrix()
    #d3 = G3.get_distance_matrix()

    # if they're all close to d1 then they're all close enough together.
    assert_array_almost_equal(d1.todense(), d2.todense(), almost_equal_decimals)
    #assert_array_almost_equal(d1.todense(), d3.todense(), almost_equal_decimals)

def test_get_distance_matrix(almost_equal_decimals = 5):
    """ test different ways to call get_distance_matrix """
    # 1. (Input type = Affinity)
    X = np.random.uniform(size = (10,10))
    Geometry = geom.Geometry(X, input_type = 'affinity')
    ## Raise error Affinity matrix passed
    msg = ( "input_method was passed as affinity. " "Distance matrix cannot be computed.")
    assert_raise_message(ValueError, msg, Geometry.get_distance_matrix)

    # 2. (No existing distance, no passed radius, no self.radius, input_type = Data)
    X = np.random.uniform(size = (10,10))
    Geometry = geom.Geometry(X, input_type = 'data')
    ## Calculate with default radius
    distance_mat = Geometry.get_distance_matrix()
    distance_mat2 = distance_matrix(X, radius = 1/X.shape[1])
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)

    # 3. (No existing distance, no passed radius, existing self.radius, input_type = Data)
    X = np.random.uniform(size = (10,10))
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius = 1)
    ## Calculate with existing radius
    distance_mat = Geometry.get_distance_matrix()
    distance_mat2 = distance_matrix(X, radius = 1)
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)

    # 4. (No existing distance, passed radius, no existing self.radius, input_type = Data)
    X = np.random.uniform(size = (10,10))
    Geometry = geom.Geometry(X, input_type = 'data')
    ## Set current radius to passed, calcualte with passed radius
    distance_mat = Geometry.get_distance_matrix(neighborhood_radius = 1)
    distance_mat2 = distance_matrix(X, radius = 1)
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)
    assert(Geometry.neighborhood_radius == 1)

    # 5. (No existing distance, passed radius equal to existing self.radius, input_type = Data)
    X = np.random.uniform(size = (10,10))
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius = 1)
    ## Calculate with passed radius
    distance_mat = Geometry.get_distance_matrix(neighborhood_radius = 1)
    distance_mat2 = distance_matrix(X, radius = 1)
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)

    # 6. (No existing distance, passed radius not equal to existing self.radius, input_type = Data)
    X = np.random.uniform(size = (10,10))
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius = 1)
    ## Calculate with passed radius, set new self.radius
    distance_mat = Geometry.get_distance_matrix(neighborhood_radius = 2)
    distance_mat2 = distance_matrix(X, radius = 2)
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)
    assert(Geometry.neighborhood_radius == 2)

    # 7. (Existing distance, no passed radius, no self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    Geometry = geom.Geometry(X, input_type = 'data')
    distance_mat2 = distance_matrix(X, radius = 1)
    Geometry.assign_distance_matrix(distance_mat2)
    ## Return existing distance matrix
    distance_mat = Geometry.get_distance_matrix()
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)

    # 8. (Existing distance, no passed radius, no self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    distance_mat2 = distance_matrix(X, radius = 1)
    Geometry = geom.Geometry(distance_mat2, input_type = 'distance')
    ## Return existing distance matrix
    distance_mat = Geometry.get_distance_matrix()
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)

    # 9. (Existing distance, no passed radius, existing self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius = 1)
    distance_mat2 = distance_matrix(X, radius = 1)
    Geometry.assign_distance_matrix(distance_mat2)
    ## Return existing distance Matrix
    distance_mat = Geometry.get_distance_matrix()
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)

    # 10. (Existing distance, no passed radius, existing self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    distance_mat2 = distance_matrix(X, radius = 1)
    Geometry = geom.Geometry(distance_mat2, input_type = 'distance', neighborhood_radius = 1)
    ## Return existing distance matrix
    distance_mat = Geometry.get_distance_matrix()
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)

    # 11. (Existing distance, passed radius, no existing self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    distance_mat2 = distance_matrix(X, radius = 1)
    Geometry = geom.Geometry(X, input_type = 'data')
    Geometry.assign_distance_matrix(distance_mat2)
    ## Re-calculate with passed radius, set self.radius to passed radius
    distance_mat = Geometry.get_distance_matrix(neighborhood_radius=3)
    distance_mat2 = distance_matrix(X, radius = 3)
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)
    assert(Geometry.neighborhood_radius == 3)

    # 12. (Existing distance, passed radius, no existing self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    distance_mat2 = distance_matrix(X, radius = 1)
    Geometry = geom.Geometry(distance_mat2, input_type = 'distance')
    ## Raise error, can't re-calculate
    msg = ("input_method was passed as distance."
           "requested radius not equal to self.neighborhood_radius."
           "distance matrix cannot be re-calculated.")
    assert_raise_message(ValueError, msg, Geometry.get_distance_matrix, neighborhood_radius=3)

    # 13. (Existing distance, passed radius equal to existing self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    distance_mat2 = distance_matrix(X, radius = 1)
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius=1)
    Geometry.assign_distance_matrix(distance_mat2)
    ## Return existing distance matrix
    distance_mat = Geometry.get_distance_matrix(neighborhood_radius=1)
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)

    # 14. (Existing distance, passed radius equal to existing self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    distance_mat2 = distance_matrix(X, radius = 1)
    Geometry = geom.Geometry(distance_mat2, input_type = 'distance', neighborhood_radius = 1)
    ## Return existing distance matrix
    distance_mat = Geometry.get_distance_matrix(neighborhood_radius=1)
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)

    # 15. (Existing distance, passed radius not equal to existing self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    distance_mat2 = distance_matrix(X, radius = 1)
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius=1)
    Geometry.assign_distance_matrix(distance_mat2)
    ## Recompute with passed radius, set self.radius to passed radius
    distance_mat2 = distance_matrix(X, radius = 3)
    distance_mat = Geometry.get_distance_matrix(neighborhood_radius=3)
    assert_array_almost_equal(distance_mat.todense(), distance_mat2.todense(), almost_equal_decimals)
    assert(Geometry.neighborhood_radius == 3)

    # 16. (Existing distance, passed radius not equal to existing self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    distance_mat2 = distance_matrix(X, radius = 1)
    Geometry = geom.Geometry(distance_mat2, input_type = 'distance', neighborhood_radius = 1)
    ## Raise error, can't recalculate
    msg = ("input_method was passed as distance."
           "requested radius not equal to self.neighborhood_radius."
           "distance matrix cannot be re-calculated.")
    assert_raise_message(ValueError, msg, Geometry.get_distance_matrix, neighborhood_radius=3)

def test_get_affinity_matrix(almost_equal_decimals=5):
    """ test different ways to call get_affinity_matrix """
    # 1. (No existing affinity, no passed radius, no self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    dist_mat = distance_matrix(X, radius = 1/X.shape[1])
    A2 = affinity_matrix(dist_mat, neighbors_radius = 1/X.shape[1])
    Geometry = geom.Geometry(X, input_type = 'data')
    ## Return default Affinity
    A = Geometry.get_affinity_matrix()
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 2. (No existing affinity, no passed radius, no self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    dist_mat = distance_matrix(X, radius = 1/X.shape[1])
    A2 = affinity_matrix(dist_mat, neighbors_radius = 1/X.shape[1])
    Geometry = geom.Geometry(dist_mat, input_type = 'distance')
    ## Return default
    A = Geometry.get_affinity_matrix()
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 3. (No existing affinity, no passed radius, existing self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius = radius,
                            affinity_radius = radius)
    ## Return with existing radius
    A = Geometry.get_affinity_matrix()
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 4. (No existing affinity, no passed radius, existing self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(dist_mat, input_type = 'distance', neighborhood_radius = radius,
                            affinity_radius = radius)
    ## Return with existing radius
    A = Geometry.get_affinity_matrix()
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 5. (No existing affinity, passed radius, no existing self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = 2)
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius = radius)
    ## Return passed radius affinity, set radius to passed
    A = Geometry.get_affinity_matrix(affinity_radius=2)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 6. (No existing affinity, passed radius, no existing self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = 2)
    Geometry = geom.Geometry(dist_mat, input_type = 'distance', neighborhood_radius = radius)
    ## Return passed radius affinity, set radius to passed
    A = Geometry.get_affinity_matrix(affinity_radius=2)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 7. (No existing affinity, passed radius equal to existing self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius = radius,
                            affinity_radius = radius)
    ## Return self.radius affinity
    A = Geometry.get_affinity_matrix(affinity_radius=radius)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 8. (No existing affinity, passed radius equal to existing self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(dist_mat, input_type = 'distance', neighborhood_radius = radius,
                            affinity_radius = radius)
    ## Return self.radius affinity
    A = Geometry.get_affinity_matrix(affinity_radius=radius)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 9. (No existing affinity, passed radius not equal to existing self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    radius = 1
    radius2 = 2
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius2)
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius = radius,
                            affinity_radius = radius)
    ## Return passed radius affinity, set radius to passed
    A = Geometry.get_affinity_matrix(affinity_radius=radius2)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)
    assert(Geometry.affinity_radius == radius2)

    # 10. (No existing affinity, passed radius not equal to existing self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    radius = 1
    radius2 = 2
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius2)
    Geometry = geom.Geometry(dist_mat, input_type = 'distance', neighborhood_radius = radius,
                            affinity_radius = radius)
    ## Return passed radius affinity, set radius to passed
    A = Geometry.get_affinity_matrix(affinity_radius=radius2)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)
    assert(Geometry.affinity_radius == radius2)

    # 11. (Existing affinity, no passed radius, no self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(X, input_type = 'data')
    Geometry.assign_affinity_matrix(A2)
    ## Return existing affinity
    A = Geometry.get_affinity_matrix()
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 12. (Existing affinity, no passed radius, no self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(dist_mat, input_type = 'distance')
    Geometry.assign_affinity_matrix(A2)
    ## Return existing affinity
    A = Geometry.get_affinity_matrix()
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 13. (Existing affinity, no passed radius, no self.radius, input_type = affinity)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(A2, input_type = 'affinity')
    ### Return existing affinity
    A = Geometry.get_affinity_matrix()
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 14. (Existing affinity, no passed radius, existing self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius = radius,
                            affinity_radius = radius)
    Geometry.assign_affinity_matrix(A2)
    ### Return existing affinity
    A = Geometry.get_affinity_matrix()
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 15. (Existing affinity, no passed radius, existing self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(dist_mat, input_type = 'distance', neighborhood_radius = radius,
                            affinity_radius = radius)
    Geometry.assign_affinity_matrix(A2)
    ### Return existing affinity
    A = Geometry.get_affinity_matrix()
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 16. (Existing affinity, no passed radius, existing self.radius, input_type = affinity)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(A2, input_type = 'affinity', neighborhood_radius = radius,
                            affinity_radius = radius)
    ## Return existing affinity
    A = Geometry.get_affinity_matrix()
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 17. (Existing affinity, passed radius, no self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = 2)
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius = radius)
    Geometry.assign_affinity_matrix(A2)
    ## Compute with passed radius, set self.radius to passed radius
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    A = Geometry.get_affinity_matrix(affinity_radius=radius)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 18. (Existing affinity, passed radius, no self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = 2)
    Geometry = geom.Geometry(dist_mat, input_type = 'distance', neighborhood_radius = radius)
    Geometry.assign_affinity_matrix(A2)
    ## Compute with passed radius, set self.radius to passed radius
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    A = Geometry.get_affinity_matrix(affinity_radius=radius)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 19. (Existing affinity, passed radius, no self.radius, input_type = affinity)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = 2)
    Geometry = geom.Geometry(A2, input_type = 'affinity', neighborhood_radius = radius)
    ## Raise error, unknown existing radius, passed radius but input type affinity
    msg = ("Input_method was passed as affinity."
           "Requested radius was not equal to self.affinity_radius."
           "Affinity Matrix cannot be recalculated.")
    assert_raise_message(ValueError, msg, Geometry.get_affinity_matrix,
                        affinity_radius=radius)

    # 20. (Existing affinity, passed radius equal to self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius = radius,
                            affinity_radius = radius)
    Geometry.assign_affinity_matrix(A2)
    ## Return existing affinity
    A = Geometry.get_affinity_matrix(affinity_radius=radius)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 21. (Existing affinity, passed radius equal to self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(dist_mat, input_type = 'distance', neighborhood_radius = radius,
                            affinity_radius = radius)
    Geometry.assign_affinity_matrix(A2)
    ## Return existing affinity
    A = Geometry.get_affinity_matrix(affinity_radius=radius)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 22. (Existing affinity, passed radius equal to self.radius, input_type = affinity)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(A2, input_type = 'affinity', neighborhood_radius = radius,
                            affinity_radius = radius)
    ## Return existing affinity
    A = Geometry.get_affinity_matrix(affinity_radius=radius)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)

    # 23. (Existing affinity, passed radius not equal to self.radius, input_type = data)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(X, input_type = 'data', neighborhood_radius = radius,
                            affinity_radius = radius)
    Geometry.assign_affinity_matrix(A2)
    ## Re-calculate with passed radius, set self.radius to passed radius
    A2 = affinity_matrix(dist_mat, neighbors_radius = 2)
    A = Geometry.get_affinity_matrix(affinity_radius=2)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)
    assert(Geometry.affinity_radius == 2)

    # 24. (Existing affinity, passed radius not equal to self.radius, input_type = distance)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = radius)
    Geometry = geom.Geometry(dist_mat, input_type = 'distance', neighborhood_radius = radius,
                            affinity_radius = radius)
    Geometry.assign_affinity_matrix(A2)
    ## Re-calculate with passed radius, set self.radius to passed radius
    A2 = affinity_matrix(dist_mat, neighbors_radius=2)
    A = Geometry.get_affinity_matrix(affinity_radius=2)
    assert_array_almost_equal(A.todense(), A2.todense(), almost_equal_decimals)
    assert(Geometry.affinity_radius == 2)

    # 25. (Existing affinity, passed radius not equal to self.radius, input_type = affinity)
    X = np.random.uniform(size = (10,10))
    radius = 1
    dist_mat = distance_matrix(X, radius = radius)
    A2 = affinity_matrix(dist_mat, neighbors_radius = 2)
    Geometry = geom.Geometry(A2, input_type = 'affinity', neighborhood_radius = radius,
                            affinity_radius = 2)
    ## Raise error, passed affinity re-calculateion not supported.
    msg = ("Input_method was passed as affinity."
           "Requested radius was not equal to self.affinity_radius."
           "Affinity Matrix cannot be recalculated.")
    assert_raise_message(ValueError, msg, Geometry.get_affinity_matrix,
                        affinity_radius=radius)

def test_get_laplacian_matrix(almost_equal_decimals = 5):
    """ test different ways to call get_laplacian_matrix """
    # 1. (No existing laplacian, no passed type, no self.type)
    X = np.random.uniform(size = (10,10))
    dist_mat = distance_matrix(X, radius = 1)
    A = affinity_matrix(dist_mat, neighbors_radius = 1)
    Geometry = geom.Geometry(A, input_type = 'affinity')
    ## Return default laplacian
    lapl = Geometry.get_laplacian_matrix()
    lapl2 = graph_laplacian(A)
    assert_array_almost_equal(lapl.todense(), lapl2.todense(),almost_equal_decimals)

    # 2. (No existing laplacian, no passed type, existing self.type)
    X = np.random.uniform(size = (10,10))
    dist_mat = distance_matrix(X, radius = 1)
    A = affinity_matrix(dist_mat, neighbors_radius = 1)
    for laplacian_type in laplacian_types:
        Geometry = geom.Geometry(A, input_type = 'affinity', laplacian_type = laplacian_type)
        ## Return self.type laplacian
        lapl = Geometry.get_laplacian_matrix()
        lapl2 = graph_laplacian(A, normed = laplacian_type)
        assert_array_almost_equal(lapl.todense(), lapl2.todense(),almost_equal_decimals)

    # 3. (No existing laplacian, passed type, no self.type)
    X = np.random.uniform(size = (10,10))
    dist_mat = distance_matrix(X, radius = 1)
    A = affinity_matrix(dist_mat, neighbors_radius = 1)
    for laplacian_type in laplacian_types:
        Geometry = geom.Geometry(A, input_type = 'affinity')
        ## Return passed type laplacian, set self.type to passed type
        lapl = Geometry.get_laplacian_matrix(laplacian_type = laplacian_type)
        lapl2 = graph_laplacian(A, normed = laplacian_type)
        assert_array_almost_equal(lapl.todense(), lapl2.todense(),almost_equal_decimals)
        assert(Geometry.laplacian_type == laplacian_type)

    # 4. (No existing laplacian, passed type equal to self.type)
    X = np.random.uniform(size = (10,10))
    dist_mat = distance_matrix(X, radius = 1)
    A = affinity_matrix(dist_mat, neighbors_radius = 1)
    for laplacian_type in laplacian_types:
        Geometry = geom.Geometry(A, input_type = 'affinity',laplacian_type = laplacian_type)
        ## Return passed type laplacian
        lapl = Geometry.get_laplacian_matrix(laplacian_type = laplacian_type)
        lapl2 = graph_laplacian(A, normed = laplacian_type)
        assert_array_almost_equal(lapl.todense(), lapl2.todense(),almost_equal_decimals)
        assert(Geometry.laplacian_type == laplacian_type)

    # 5. (No existing laplacian, passed type not equal to self.type)
    X = np.random.uniform(size = (10,10))
    dist_mat = distance_matrix(X, radius = 1)
    A = affinity_matrix(dist_mat, neighbors_radius = 1)
    for laplacian_type in laplacian_types:
        if laplacian_type == 'geometric':
            existing_type = 'randomwalk'
        else:
            existing_type = 'geometric'
        Geometry = geom.Geometry(A, input_type = 'affinity', laplacian_type = existing_type)
        ## Set type to passed type, return passed type laplacian
        lapl = Geometry.get_laplacian_matrix(laplacian_type = laplacian_type)
        lapl2 = graph_laplacian(A, normed = laplacian_type)
        assert_array_almost_equal(lapl.todense(), lapl2.todense(),almost_equal_decimals)
        assert(Geometry.laplacian_type == laplacian_type)

    # 6. (Existing laplacian, no passed type, no self.type)
    X = np.random.uniform(size = (10,10))
    dist_mat = distance_matrix(X, radius = 1)
    A = affinity_matrix(dist_mat, neighbors_radius = 1)
    Geometry = geom.Geometry(A, input_type = 'affinity')
    lapl2 = graph_laplacian(A)
    Geometry.assign_laplacian_matrix(lapl2)
    ## Return existing laplacian
    lapl = Geometry.get_laplacian_matrix()
    assert_array_almost_equal(lapl.todense(), lapl2.todense(),almost_equal_decimals)

    # 7. (Existing laplacian, no passed type, existing self.type)
    X = np.random.uniform(size = (10,10))
    dist_mat = distance_matrix(X, radius = 1)
    A = affinity_matrix(dist_mat, neighbors_radius = 1)
    Geometry = geom.Geometry(A, input_type = 'affinity', laplacian_type = 'geometric')
    lapl2 = graph_laplacian(A)
    Geometry.assign_laplacian_matrix(lapl2)
    ## Return existing laplacian
    lapl = Geometry.get_laplacian_matrix()
    assert_array_almost_equal(lapl.todense(), lapl2.todense(),almost_equal_decimals)

    # 8. (Existing laplacian, passed type, no self.type)
    X = np.random.uniform(size = (10,10))
    dist_mat = distance_matrix(X, radius = 1)
    A = affinity_matrix(dist_mat, neighbors_radius = 1)
    Geometry = geom.Geometry(A, input_type = 'affinity')
    lapl2 = graph_laplacian(A)
    Geometry.assign_laplacian_matrix(lapl2)
    ## Calculate passed type, set self.type to passed type
    # this will warn that it will overwrite:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lapl = Geometry.get_laplacian_matrix(laplacian_type = 'randomwalk')
    lapl2 = graph_laplacian(A, normed = 'randomwalk')
    assert_array_almost_equal(lapl.todense(), lapl2.todense(),almost_equal_decimals)

    # 9. (Existing laplacian, passed type equal to self.type)
    X = np.random.uniform(size = (10,10))
    dist_mat = distance_matrix(X, radius = 1)
    A = affinity_matrix(dist_mat, neighbors_radius = 1)
    for laplacian_type in laplacian_types:
        Geometry = geom.Geometry(A, input_type = 'affinity', laplacian_type = laplacian_type)
        lapl2 = graph_laplacian(A, normed = laplacian_type)
        Geometry.assign_laplacian_matrix(lapl2, laplacian_type = laplacian_type)
        ## Return existing laplacian
        lapl = Geometry.get_laplacian_matrix(laplacian_type = laplacian_type)
        assert_array_almost_equal(lapl.todense(), lapl2.todense(),almost_equal_decimals)
        assert(Geometry.laplacian_type == laplacian_type)

    # 10. (Existing laplacian, passed type not equal to self.type)
    X = np.random.uniform(size = (10,10))
    dist_mat = distance_matrix(X, radius = 1)
    A = affinity_matrix(dist_mat, neighbors_radius = 1)
    for laplacian_type in laplacian_types:
        if laplacian_type == 'geometric':
            existing_type = 'randomwalk'
        else:
            existing_type = 'geometric'
        Geometry = geom.Geometry(A, input_type = 'affinity', laplacian_type = existing_type)
        lapl2 = graph_laplacian(A, normed = existing_type)
        Geometry.assign_laplacian_matrix(lapl2, laplacian_type = existing_type)
        ## Calculate passed type, set self.type to passed type
        # this will warn that it will overwrite:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lapl = Geometry.get_laplacian_matrix(laplacian_type = laplacian_type)
        lapl2 = graph_laplacian(A, normed = laplacian_type)
        assert_array_almost_equal(lapl.todense(), lapl2.todense(),almost_equal_decimals)
        assert(Geometry.laplacian_type == laplacian_type)
