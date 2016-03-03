from nose import SkipTest

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import isspmatrix

from megaman.geometry.distance_new import adjacency_graph, Adjacency

try:
    import pyflann as pyf
    NO_PYFLANN = False
except ImportError:
    NO_PYFLANN = True


def test_adjacency():
    X = np.random.rand(100, 3)
    Gtrue = {}

    assert len(Adjacency.methods()) == 5

    exact_methods = [m for m in Adjacency.methods()
                     if not m.endswith('flann')]

    def check_kneighbors(n_neighbors, method, exact=True):
        if method == 'pyflann' and NO_PYFLANN:
            raise SkipTest("pyflann not installed")

        G = adjacency_graph(X, method=method,
                            n_neighbors=n_neighbors)
        assert isspmatrix(G)
        assert G.shape == (X.shape[0], X.shape[0])
        if method in exact_methods:
            assert_allclose(G.toarray(), Gtrue[n_neighbors].toarray())

    def check_radius(radius, method, exact=True):
        if method == 'pyflann' and NO_PYFLANN:
            raise SkipTest("pyflann not installed")

        G = adjacency_graph(X, method=method,
                            radius=radius)
        assert isspmatrix(G)
        assert G.shape == (X.shape[0], X.shape[0])
        if method in exact_methods:
            assert_allclose(G.toarray(), Gtrue[radius].toarray())

    for n_neighbors in [5, 10, 15]:
        Gtrue[n_neighbors] = adjacency_graph(X, method='brute',
                                             n_neighbors=n_neighbors)
        for method in Adjacency.methods():
            yield check_kneighbors, n_neighbors, method

    for radius in [0.1, 0.5, 1.0]:
        Gtrue[radius] = adjacency_graph(X, method='brute',
                                        radius=radius)
        for method in Adjacency.methods():
            yield check_radius, radius, method
