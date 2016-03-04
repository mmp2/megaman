import numpy as np
from numpy.testing import assert_allclose, assert_equal

from scipy.spatial.distance import cdist

from megaman.geometry.adjacency import compute_adjacency_matrix
from megaman.geometry.affinity import compute_affinity_matrix, Affinity


def test_affinity():
    rand = np.random.RandomState(42)
    X = np.random.rand(20, 3)
    D = cdist(X, X)

    def check_affinity(adjacency_radius, affinity_radius, symmetrize):
        adj = compute_adjacency_matrix(X, radius=adjacency_radius)
        aff = compute_affinity_matrix(adj, radius=affinity_radius,
                                      symmetrize=True)

        A = np.exp(-(D / affinity_radius) ** 2)
        A[D > adjacency_radius] = 0
        assert_allclose(aff.toarray(), A)

    for adjacency_radius in [0.5, 1.0, 5.0]:
        for affinity_radius in [0.1, 0.5, 1.0]:
            for symmetrize in [True, False]:
                yield (check_affinity, adjacency_radius,
                       affinity_radius, symmetrize)


def test_custom_affinity():
    class CustomAffinity(Affinity):
        name = "custom"
        def affinity_matrix(self, adjacency_matrix):
            return np.exp(-abs(adjacency_matrix.toarray()))

    rand = np.random.RandomState(42)
    X = rand.rand(10, 2)
    D = compute_adjacency_matrix(X, radius=10)
    A = compute_affinity_matrix(D, method='custom', radius=1)
    assert_allclose(A, np.exp(-abs(D.toarray())))

    Affinity._remove_from_registry("custom")
