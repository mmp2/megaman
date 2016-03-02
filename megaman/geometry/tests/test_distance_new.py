import numpy as np
from numpy.testing import assert_allclose


from megaman.geometry.distance_new import Adjacency, BruteForceAdjacency


def test_adjacency():
    X = np.random.rand(100, 3)
    Gtrue = {}

    assert len(Adjacency.methods()) == 3

    def check_kneighbors(n_neighbors, method):
        Estimator = Adjacency.get_method(method)
        G = Estimator(n_neighbors=n_neighbors).adjacency_graph(X)
        assert_allclose(G.toarray(), Gtrue[n_neighbors].toarray())

    def check_radius(radius, method):
        Estimator = Adjacency.get_method(method)
        G = Estimator(radius=radius).adjacency_graph(X)
        assert_allclose(G.toarray(), Gtrue[radius].toarray())

    for n_neighbors in [5, 10, 15]:
        Gtrue[n_neighbors] = BruteForceAdjacency(n_neighbors=n_neighbors).adjacency_graph(X)
        for method in Adjacency.methods():
            yield check_kneighbors, n_neighbors, method

    for radius in [0.1, 0.5, 1.0]:
        Gtrue[radius] = BruteForceAdjacency(radius=radius).adjacency_graph(X)
        for method in Adjacency.methods():
            yield check_radius, radius, method
