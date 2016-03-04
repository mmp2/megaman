from nose import SkipTest

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import isspmatrix
from scipy.spatial.distance import cdist, pdist, squareform

from megaman.geometry.distance import compute_adjacency_matrix, Adjacency


try:
    import pyflann as pyf
    NO_PYFLANN = False
except ImportError:
    NO_PYFLANN = True


def test_adjacency():
    X = np.random.rand(100, 3)
    Gtrue = {}

    exact_methods = [m for m in Adjacency.methods()
                     if not m.endswith('flann')]

    def check_kneighbors(n_neighbors, method, exact=True):
        if method == 'pyflann' and NO_PYFLANN:
            raise SkipTest("pyflann not installed")

        G = compute_adjacency_matrix(X, method=method,
                            n_neighbors=n_neighbors)
        assert isspmatrix(G)
        assert G.shape == (X.shape[0], X.shape[0])
        if method in exact_methods:
            assert_allclose(G.toarray(), Gtrue[n_neighbors].toarray())

    def check_radius(radius, method, exact=True):
        if method == 'pyflann' and NO_PYFLANN:
            raise SkipTest("pyflann not installed")

        G = compute_adjacency_matrix(X, method=method,
                            radius=radius)
        assert isspmatrix(G)
        assert G.shape == (X.shape[0], X.shape[0])
        if method in exact_methods:
            assert_allclose(G.toarray(), Gtrue[radius].toarray())

    for n_neighbors in [5, 10, 15]:
        Gtrue[n_neighbors] = compute_adjacency_matrix(X, method='brute',
                                             n_neighbors=n_neighbors)
        for method in Adjacency.methods():
            yield check_kneighbors, n_neighbors, method

    for radius in [0.1, 0.5, 1.0]:
        Gtrue[radius] = compute_adjacency_matrix(X, method='brute',
                                        radius=radius)
        for method in Adjacency.methods():
            yield check_radius, radius, method


def test_new_adjacency():
    class TestAdjacency(Adjacency):
        name = "_test"
        def adjacency_graph(self, X):
            return squareform(pdist(X))

    rand = np.random.RandomState(42)
    X = rand.rand(10, 2)
    D = compute_adjacency_matrix(X, method='_test', radius=1)
    assert_allclose(D, cdist(X, X))
