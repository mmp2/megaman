import numpy as np
from numpy.testing import assert_allclose, assert_equal

from scipy.sparse import isspmatrix

from megaman.geometry.adjacency_new import compute_adjacency_matrix
from megaman.geometry.affinity_new import compute_affinity_matrix
from megaman.geometry.laplacian_new import Laplacian, compute_laplacian_matrix


def test_laplacian_smoketest():
    rand = np.random.RandomState(42)
    X = rand.rand(20, 2)
    adj = compute_adjacency_matrix(X, radius=0.5)
    aff = compute_affinity_matrix(adj, radius=0.1)

    def check_laplacian(method):
        lap = compute_laplacian_matrix(aff, method=method)

        assert isspmatrix(lap)
        assert_equal(lap.shape, (X.shape[0], X.shape[0]))

    for method in Laplacian.asymmetric_methods():
        yield check_laplacian, method


def test_laplacian_full_output():
    # Test that full_output symmetrized laplacians have the right form
    rand = np.random.RandomState(42)
    X = rand.rand(20, 2)

    def check_symmetric(method, adjacency_radius, affinity_radius):
        adj = compute_adjacency_matrix(X, radius=adjacency_radius)
        aff = compute_affinity_matrix(adj, radius=affinity_radius)
        lap, lapsym, w = compute_laplacian_matrix(aff, method=method,
                                                  full_output=True)

        sym = w[:, np.newaxis] * (lap.toarray() + np.eye(*lap.shape))

        assert_allclose(lapsym.toarray(), sym)

    for method in Laplacian.asymmetric_methods():
        for adjacency_radius in [0.5, 1.0]:
            for affinity_radius in [0.1, 0.3]:
                yield check_symmetric, method, adjacency_radius, affinity_radius
