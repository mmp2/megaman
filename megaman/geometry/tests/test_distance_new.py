import numpy as np
from numpy.testing import assert_allclose


from megaman.geometry.distance_new import adjacency_graph, Adjacency


def test_adjacency():
    X = np.random.rand(100, 3)
    Gtrue = {}

    assert len(Adjacency.methods()) == 3

    def check_kneighbors(n_neighbors, method):
        G = adjacency_graph(X, method=method,
                            n_neighbors=n_neighbors)
        assert_allclose(G.toarray(), Gtrue[n_neighbors].toarray())

    def check_radius(radius, method):
        G = adjacency_graph(X, method=method,
                            radius=radius)
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
