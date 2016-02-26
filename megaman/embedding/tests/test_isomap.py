import sys
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.spatial.distance import squareform, pdist
from itertools import product

from numpy.testing import assert_array_almost_equal

import megaman.embedding.isomap as iso
import megaman.geometry.geometry as geom

eigen_solvers = ['auto', 'dense', 'arpack', 'amg', 'lobpcg']

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

def test_isomap_with_sklearn():
    try:
        from sklearn import manifold
        from sklearn import datasets
        from sklearn.neighbors import NearestNeighbors
        N = 10
        X, color = datasets.samples_generator.make_s_curve(N, random_state=0)
        n_components = 2
        n_neighbors = 3
        knn = NearestNeighbors(n_neighbors + 1).fit(X)
        # Assign the geometry matrix to get the same answer since sklearn using k-neighbors instead of radius-neighbors
        Geometry = geom.Geometry(X)
        Geometry.assign_distance_matrix(knn.kneighbors_graph(X, mode = 'distance'))
        # test Isomap with sklearn
        sk_Y_iso = manifold.Isomap(n_neighbors, n_components, eigen_solver = 'arpack').fit_transform(X)
        mm_Y_iso = iso.isomap(Geometry, n_components)
        assert(_check_with_col_sign_flipping(sk_Y_iso, mm_Y_iso, 0.05))
    except ImportError:
        return True

def test_isomap_simple_grid():
    # Isomap should preserve distances when all neighbors are used
    N_per_side = 5
    Npts = N_per_side ** 2
    radius = 10

    # grid of equidistant points in 2D, n_components = n_dim
    X = np.array(list(product(range(N_per_side), repeat=2)))

    # distances from each point to all others
    G = squareform(pdist(X))
    Geometry = geom.Geometry(X, neighborhood_radius = radius)

    for eigen_solver in eigen_solvers:
        clf = iso.Isomap(n_components = 2, eigen_solver = eigen_solver,
                        Geometry = Geometry)
        clf.fit(X)
        G_iso = squareform(pdist(clf.embedding_))
        assert_array_almost_equal(G, G_iso)
