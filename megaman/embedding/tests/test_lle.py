import sys
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.spatial.distance import squareform, pdist
from itertools import product
from numpy.testing import assert_array_almost_equal

import megaman.embedding.locally_linear as lle
import megaman.geometry.geometry as geom

eigen_solvers = ['auto', 'dense', 'amg', 'lobpcg', 'arpack']

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
    sk_Y_lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components, method = 'standard').fit_transform(X)
    (mm_Y_lle, err) = lle.locally_linear_embedding(Geometry, n_components)
    assert(_check_with_col_sign_flipping(sk_Y_lle, mm_Y_lle, 0.05))

def test_barycenter_kneighbors_graph():
    X = np.array([[0, 1], [1.01, 1.], [2, 0]])
    distance_matrix = squareform(pdist(X))
    A = lle.barycenter_graph(distance_matrix, X)
    # check that columns sum to one
    assert_array_almost_equal(np.sum(A.toarray(), 1), np.ones(3))
    pred = np.dot(A.toarray(), X)
    assert(np.linalg.norm(pred - X) / X.shape[0] < 1)

def test_lle_simple_grid():
    # note: ARPACK is numerically unstable, so this test will fail for
    #       some random seeds.  We choose 20 because the tests pass.
    rng = np.random.RandomState(20)
    tol = 0.1
    # grid of equidistant points in 2D, n_components = n_dim
    X = np.array(list(product(range(5), repeat=2)))
    X = X + 1e-10 * rng.uniform(size=X.shape)
    n_components = 2
    Geometry = geom.Geometry(X, neighborhood_radius = 3)
    tol = 0.1
    distance_matrix = Geometry.get_distance_matrix()
    N = lle.barycenter_graph(distance_matrix, X).todense()
    reconstruction_error = np.linalg.norm(np.dot(N, X) - X, 'fro')
    assert(reconstruction_error < tol)
    for eigen_solver in eigen_solvers:
        clf = lle.LocallyLinearEmbedding(n_components = n_components, Geometry = Geometry,
                                eigen_solver = eigen_solver, random_state = rng)
        clf.fit(X)
        assert(clf.embedding_.shape[1] == n_components)
        reconstruction_error = np.linalg.norm(
        np.dot(N, clf.embedding_) - clf.embedding_, 'fro') ** 2
        assert(reconstruction_error < tol)

def test_lle_manifold():
    rng = np.random.RandomState(0)
    # similar test on a slightly more complex manifold
    X = np.array(list(product(np.arange(18), repeat=2)))
    X = np.c_[X, X[:, 0] ** 2 / 18]
    X = X + 1e-10 * rng.uniform(size=X.shape)
    n_components = 2
    Geometry = geom.Geometry(X, neighborhood_radius = 3)
    distance_matrix = Geometry.get_distance_matrix()
    tol = 1.5
    N = lle.barycenter_graph(distance_matrix, X).todense()
    reconstruction_error = np.linalg.norm(np.dot(N, X) - X)
    assert(reconstruction_error < tol)
    for eigen_solver in eigen_solvers:
        clf = lle.LocallyLinearEmbedding(n_components = n_components, Geometry = Geometry,
                                eigen_solver = eigen_solver, random_state = rng)
        clf.fit(X)
        assert(clf.embedding_.shape[1] == n_components)
        reconstruction_error = np.linalg.norm(
            np.dot(N, clf.embedding_) - clf.embedding_, 'fro') ** 2
        assert(reconstruction_error < tol)
