# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import sys
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from itertools import product

from sklearn import manifold, datasets
from sklearn.neighbors import NearestNeighbors

from numpy.testing import assert_array_almost_equal
import megaman.embedding.ltsa as ltsa
from megaman.embedding.locally_linear import barycenter_graph
import megaman.geometry.geometry as geom
from megaman.utils.eigendecomp import EIGEN_SOLVERS


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


def test_ltsa_with_sklearn():
    N = 10
    X, color = datasets.samples_generator.make_s_curve(N, random_state=0)
    n_components = 2
    n_neighbors = 3
    knn = NearestNeighbors(n_neighbors + 1).fit(X)
    G = geom.Geometry()
    G.set_data_matrix(X)
    G.set_adjacency_matrix(knn.kneighbors_graph(X, mode = 'distance'))
    sk_Y_ltsa = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                                method = 'ltsa',
                                                eigen_solver = 'arpack').fit_transform(X)
    (mm_Y_ltsa, err) = ltsa.ltsa(G, n_components, eigen_solver = 'arpack')
    assert(_check_with_col_sign_flipping(sk_Y_ltsa, mm_Y_ltsa, 0.05))


def test_ltsa_eigendecomps():
    N = 10
    X, color = datasets.samples_generator.make_s_curve(N, random_state=0)
    n_components = 2
    G = geom.Geometry(adjacency_method = 'brute', adjacency_kwds = {'radius':2})
    G.set_data_matrix(X)
    mm_ltsa_ref, err_ref = ltsa.ltsa(G, n_components,
                                     eigen_solver=EIGEN_SOLVERS[0])
    for eigen_solver in EIGEN_SOLVERS[1:]:
        mm_ltsa, err = ltsa.ltsa(G, n_components, eigen_solver=eigen_solver)
        assert(_check_with_col_sign_flipping(mm_ltsa, mm_ltsa_ref, 0.05))


def test_ltsa_manifold():
    rng = np.random.RandomState(0)
    # similar test on a slightly more complex manifold
    X = np.array(list(product(np.arange(18), repeat=2)))
    X = np.c_[X, X[:, 0] ** 2 / 18]
    X = X + 1e-10 * rng.uniform(size=X.shape)
    n_components = 2
    G = geom.Geometry(adjacency_kwds = {'radius':3})
    G.set_data_matrix(X)
    distance_matrix = G.compute_adjacency_matrix()
    tol = 1.5
    N = barycenter_graph(distance_matrix, X).todense()
    reconstruction_error = np.linalg.norm(np.dot(N, X) - X)
    assert(reconstruction_error < tol)
    for eigen_solver in EIGEN_SOLVERS:
        clf = ltsa.LTSA(n_components = n_components, geom = G,
                        eigen_solver = eigen_solver, random_state = rng)
        clf.fit(X)
        assert(clf.embedding_.shape[1] == n_components)
        reconstruction_error = np.linalg.norm(
            np.dot(N, clf.embedding_) - clf.embedding_, 'fro') ** 2
        assert(reconstruction_error < tol)
