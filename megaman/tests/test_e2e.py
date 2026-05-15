"""
End-to-end integration tests for the megaman manifold learning pipeline.

These tests exercise the full workflow:
  1. Generate synthetic data (S-curve, swiss roll, blobs)
  2. Compute geometry (adjacency → affinity → laplacian)
  3. Run embedding algorithms (Isomap, LLE, LTSA, SpectralEmbedding)
  4. Validate output shapes, types, and basic correctness
  5. Test fit / transform / predict round-trip
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import issparse
from sklearn.datasets import make_s_curve, make_swiss_roll, make_blobs


# ---------------------------------------------------------------------------
# Geometry pipeline E2E
# ---------------------------------------------------------------------------

class TestGeometryPipeline:
    """Full geometry computation pipeline: adjacency → affinity → laplacian."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(200, 3)
        self.radius = 1.5

    def test_full_geometry_object(self):
        """Geometry object should lazily compute all three matrices."""
        from megaman.geometry import Geometry

        geom = Geometry(
            adjacency_method='cyflann',
            adjacency_kwds={'radius': self.radius},
            affinity_kwds={'radius': self.radius},
            laplacian_method='geometric',
        )
        geom.set_data_matrix(self.X)

        # Adjacency
        adj = geom.compute_adjacency_matrix()
        assert issparse(adj)
        assert adj.shape == (200, 200)

        # Affinity
        aff = geom.compute_affinity_matrix()
        assert issparse(aff)
        assert aff.shape == (200, 200)
        # affinity values should be in [0, 1]
        assert aff.data.min() >= 0
        assert aff.data.max() <= 1.0 + 1e-10

        # Laplacian
        lap = geom.compute_laplacian_matrix()
        assert issparse(lap)
        assert lap.shape == (200, 200)

    def test_adjacency_methods_agree(self):
        """brute and cyflann should give similar adjacency matrices."""
        from megaman.geometry import compute_adjacency_matrix

        adj_brute = compute_adjacency_matrix(self.X, method='brute', radius=self.radius)
        adj_cyflann = compute_adjacency_matrix(self.X, method='cyflann', radius=self.radius)

        # cyflann is approximate, so check structural similarity
        brute_nnz = adj_brute.nnz
        cyflann_nnz = adj_cyflann.nnz
        # Should find a similar number of neighbors (within 50% for approximate)
        assert abs(brute_nnz - cyflann_nnz) / max(brute_nnz, 1) < 0.5

    def test_laplacian_methods_all_run(self):
        """All laplacian methods should run without error."""
        from megaman.geometry import (compute_adjacency_matrix,
                                       compute_affinity_matrix,
                                       compute_laplacian_matrix,
                                       laplacian_methods)

        adj = compute_adjacency_matrix(self.X, radius=self.radius)
        aff = compute_affinity_matrix(adj, radius=self.radius)

        for method in laplacian_methods():
            if method == 'auto':
                continue
            kwargs = {'scaling_epps': self.radius}
            if method == 'renormalized':
                kwargs['renormalization_exponent'] = 1.5
            lap = compute_laplacian_matrix(aff, method=method, **kwargs)
            assert issparse(lap), f"Laplacian method '{method}' did not return sparse matrix"
            assert lap.shape == (200, 200)


# ---------------------------------------------------------------------------
# Embedding pipeline E2E
# ---------------------------------------------------------------------------

class TestEmbeddingPipeline:
    """Full embedding pipeline: data → geometry → embedding → validation."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.N = 300
        self.n_components = 2
        self.X, _ = make_s_curve(self.N, random_state=0)
        self.radius = 2.0
        self.geom_params = {
            'adjacency_method': 'cyflann',
            'adjacency_kwds': {'radius': self.radius},
            'affinity_kwds': {'radius': self.radius},
            'laplacian_method': 'geometric',
        }

    def test_isomap_e2e(self):
        """Isomap should produce a valid low-dimensional embedding."""
        from megaman.embedding import Isomap

        model = Isomap(
            n_components=self.n_components,
            geom=self.geom_params,
            radius=self.radius,
            random_state=42,
        )
        embedding = model.fit_transform(self.X)

        assert embedding.shape == (self.N, self.n_components)
        assert np.isfinite(embedding).all()
        assert_allclose(embedding, model.embedding_)

    def test_lle_e2e(self):
        """Locally Linear Embedding should produce a valid embedding."""
        from megaman.embedding import LocallyLinearEmbedding

        model = LocallyLinearEmbedding(
            n_components=self.n_components,
            geom=self.geom_params,
            radius=self.radius,
            random_state=42,
        )
        embedding = model.fit_transform(self.X)

        assert embedding.shape == (self.N, self.n_components)
        assert np.isfinite(embedding).all()

    def test_ltsa_e2e(self):
        """LTSA should produce a valid embedding."""
        from megaman.embedding import LTSA

        model = LTSA(
            n_components=self.n_components,
            geom=self.geom_params,
            radius=self.radius,
            random_state=42,
        )
        embedding = model.fit_transform(self.X)

        assert embedding.shape == (self.N, self.n_components)
        assert np.isfinite(embedding).all()

    def test_spectral_embedding_e2e(self):
        """Spectral Embedding should produce a valid embedding."""
        from megaman.embedding import SpectralEmbedding

        model = SpectralEmbedding(
            n_components=self.n_components,
            geom=self.geom_params,
            eigen_solver='arpack',
            random_state=42,
        )
        embedding = model.fit_transform(self.X)

        assert embedding.shape == (self.N, self.n_components)
        assert np.isfinite(embedding).all()

    def test_spectral_embedding_predict(self):
        """SpectralEmbedding.predict() should embed new points."""
        from megaman.embedding import SpectralEmbedding

        geom_params = {
            'adjacency_method': 'brute',
            'adjacency_kwds': {'radius': self.radius},
            'affinity_kwds': {'radius': self.radius},
            'laplacian_method': 'geometric',
        }

        model = SpectralEmbedding(
            n_components=self.n_components,
            geom=geom_params,
            eigen_solver='arpack',
            random_state=42,
        )

        X_train = self.X[:250]
        X_test = self.X[250:]

        model.fit(X_train)
        embed_test, embed_all = model.predict(X_test)

        assert embed_test.shape == (50, self.n_components)
        assert embed_all.shape == (self.N, self.n_components)
        assert np.isfinite(embed_test).all()
        assert np.isfinite(embed_all).all()

    def test_fit_vs_fit_transform_consistency(self):
        """fit() + embedding_ should match fit_transform() output."""
        from megaman.embedding import SpectralEmbedding

        model = SpectralEmbedding(
            n_components=self.n_components,
            geom=self.geom_params,
            eigen_solver='arpack',
            random_state=42,
        )

        result = model.fit_transform(self.X)
        assert_allclose(result, model.embedding_)


# ---------------------------------------------------------------------------
# Eigendecomposition E2E
# ---------------------------------------------------------------------------

class TestEigendecompPipeline:
    """Test that all eigen solvers produce consistent results."""

    def test_all_solvers_agree(self):
        from megaman.utils.eigendecomp import eigen_decomposition, EIGEN_SOLVERS

        rng = np.random.RandomState(0)
        X = rng.uniform(size=(100, 40))
        S = X.T @ X  # symmetric positive definite

        results = {}
        for solver in EIGEN_SOLVERS:
            lambdas, vectors = eigen_decomposition(
                S, n_components=3, eigen_solver=solver,
                largest=True, drop_first=False,
            )
            results[solver] = np.sort(lambdas)

        # All solvers should agree on eigenvalues
        ref = results['dense']
        for solver, vals in results.items():
            assert_allclose(vals, ref, atol=1e-5,
                            err_msg=f"{solver} disagrees with dense solver")


# ---------------------------------------------------------------------------
# cyflann E2E
# ---------------------------------------------------------------------------

class TestCyflannPipeline:
    """Test the FLANN nearest-neighbor index end-to-end."""

    def test_knn_search(self):
        """cyflann Index should build and query successfully."""
        from megaman.geometry.cyflann.index import Index

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3).astype(np.float64)

        idx = Index(X)
        idx.buildIndex()
        # verify the index knows its data dimensions
        assert idx.size() == 100
        assert idx.veclen() == 3

    def test_radius_search_consistency(self):
        """cyflann radius search should find the query point itself."""
        from megaman.geometry import compute_adjacency_matrix

        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)

        adj = compute_adjacency_matrix(X, method='cyflann', radius=2.0)
        assert issparse(adj)
        # Every point should have at least one neighbor (itself or nearby)
        assert adj.nnz > 0


# ---------------------------------------------------------------------------
# Swiss roll dataset — stress test
# ---------------------------------------------------------------------------

class TestSwissRollStress:
    """Larger-scale test with swiss roll data."""

    def test_swiss_roll_spectral(self):
        X, color = make_swiss_roll(n_samples=500, random_state=0)
        radius = 3.0
        geom_params = {
            'adjacency_method': 'cyflann',
            'adjacency_kwds': {'radius': radius},
            'affinity_kwds': {'radius': radius},
            'laplacian_method': 'geometric',
        }

        from megaman.embedding import SpectralEmbedding
        model = SpectralEmbedding(
            n_components=2,
            geom=geom_params,
            eigen_solver='arpack',
            random_state=42,
        )
        embedding = model.fit_transform(X)

        assert embedding.shape == (500, 2)
        assert np.isfinite(embedding).all()
        # Embedding should have non-trivial variance in both components
        assert embedding.std(axis=0).min() > 1e-6


# ---------------------------------------------------------------------------
# Clustering E2E (spectral clustering)
# ---------------------------------------------------------------------------

class TestSpectralClusteringE2E:
    """Test spectral clustering produces valid labels via the full pipeline."""

    def test_clustering_separates_clusters(self):
        """SpectralClustering should label well-separated clusters correctly."""
        from megaman.utils.spectral_clustering import SpectralClustering

        K = 2
        num_per_cluster = 50
        # Two well-separated unit-vector clusters
        c = np.array([[1, 0, 0], [0, 1, 0]])
        X = np.repeat(c, num_per_cluster, axis=0).astype(np.float64)

        SC = SpectralClustering(
            K=K, radius=5, stabalize=True, renormalize=True,
            eigen_solver='dense', random_state=42,
        )
        labels = SC.fit_transform(X, input_type='data')

        assert labels.shape == (K * num_per_cluster,)
        assert set(labels) == {0, 1}
        # All points in each cluster should have the same label
        cluster_0 = labels[:num_per_cluster]
        cluster_1 = labels[num_per_cluster:]
        assert np.all(cluster_0 == cluster_0[0])
        assert np.all(cluster_1 == cluster_1[0])
        assert cluster_0[0] != cluster_1[0]
