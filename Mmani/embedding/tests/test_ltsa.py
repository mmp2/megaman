import sys
import numpy as np
import scipy as sp
import scipy.sparse as sparse

from numpy.testing import assert_array_almost_equal

import Mmani.embedding.ltsa as ltsa
import Mmani.geometry.geometry as geom


def _check_with_col_sign_flipping(A, B, tol=0.0):
    """ Check array A and B are equal with possible sign flipping on
    each columns"""
    sign = True
    for column_idx in range(A.shape[1]):
        sign = sign and ((((A[:, column_idx] -
                            B[:, column_idx]) ** 2).mean() <= tol ** 2) or
                         (((A[:, column_idx] +
                            B[:, column_idx]) ** 2).mean() <= tol ** 2))
        if not sign:
            return False
    return True

def test_lle_with_sklearn():
    from sklearn import manifold
    from sklearn import datasets
    from sklearn.neighbors import NearestNeighbors
    N = 10
    X, color = datasets.samples_generator.make_s_curve(N, random_state=0)
    n_components = 2
    n_neighbors = 3
    knn = NearestNeighbors(n_neighbors + 1).fit(X)
    Geometry = geom.Geometry(X)
    Geometry.assign_distance_matrix(knn.kneighbors_graph(X, mode = 'distance'))
    sk_Y_ltsa = manifold.LocallyLinearEmbedding(n_neighbors, n_components, 
                                                method = 'ltsa',
                                                eigen_solver = 'arpack').fit_transform(X)
    (mm_Y_ltsa, err) = ltsa.ltsa(Geometry, n_components, eigen_solver = 'arpack')
    assert(_check_with_col_sign_flipping(sk_Y_ltsa, mm_Y_ltsa, 0.05))