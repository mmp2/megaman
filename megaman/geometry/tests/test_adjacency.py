# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

from nose import SkipTest

import numpy as np
from numpy.testing import assert_allclose, assert_raises, assert_equal
from scipy.sparse import isspmatrix
from scipy.spatial.distance import cdist, pdist, squareform

from megaman.geometry import (Geometry, compute_adjacency_matrix, Adjacency,
                              adjacency_methods)


try:
    import pyflann as pyf
    NO_PYFLANN = False
except ImportError:
    NO_PYFLANN = True


def test_adjacency_methods():
    assert_equal(set(adjacency_methods()),
                 {'auto', 'pyflann', 'ball_tree',
                  'cyflann', 'brute', 'kd_tree'})


def test_adjacency_input_validation():
    X = np.random.rand(20, 3)
    # need to specify radius or n_neighbors
    assert_raises(ValueError, compute_adjacency_matrix, X)
    # cannot specify both radius and n_neighbors
    assert_raises(ValueError, compute_adjacency_matrix, X,
                  radius=1, n_neighbors=10)


def test_adjacency():
    rng = np.random.RandomState(36)
    X = rng.rand(100, 3)
    Gtrue = {}

    exact_methods = [m for m in Adjacency.methods()
                     if not m.endswith('flann')]

    def check_kneighbors(n_neighbors, method):
        if method == 'pyflann' and NO_PYFLANN:
            raise SkipTest("pyflann not installed")

        G = compute_adjacency_matrix(X, method=method,
                            n_neighbors=n_neighbors)
        assert isspmatrix(G)
        assert G.shape == (X.shape[0], X.shape[0])
        if method in exact_methods:
            assert_allclose(G.toarray(), Gtrue[n_neighbors].toarray())

    def check_radius(radius, method):
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


def test_unknown_method():
    X = np.arange(20).reshape((10, 2))
    assert_raises(ValueError, compute_adjacency_matrix, X, 'foo')


def test_all_methods_close():
    rand = np.random.RandomState(36)
    X = rand.randn(10, 2)
    D_true = squareform(pdist(X))
    D_true[D_true > 0.5] = 0

    def check_method(method):
        kwargs = {}
        if method == 'pyflann':
            try:
                import pyflann as pyf
            except ImportError:
                raise SkipTest("pyflann not installed.")
            flindex = pyf.FLANN()
            flindex.build_index(X, algorithm='kmeans',
                                target_precision=0.9)
            kwargs['flann_index'] = flindex
        this_D = compute_adjacency_matrix(X, method=method, radius=0.5,
                                          **kwargs)
        assert_allclose(this_D.toarray(), D_true, rtol=1E-5)

    for method in ['auto', 'cyflann', 'pyflann', 'brute']:
        yield check_method, method


def test_custom_adjacency():
    class CustomAdjacency(Adjacency):
        name = "custom"
        def adjacency_graph(self, X):
            return squareform(pdist(X))

    rand = np.random.RandomState(42)
    X = rand.rand(10, 2)
    D = compute_adjacency_matrix(X, method='custom', radius=1)
    assert_allclose(D, cdist(X, X))

    Adjacency._remove_from_registry("custom")

def test_cyflann_index_type():
    rand = np.random.RandomState(36)
    X = rand.randn(10, 2)
    D_true = squareform(pdist(X))
    D_true[D_true > 1.5] = 0
    
    def check_index_type(index_type):
        method = 'cyflann'
        radius = 1.5
        cyflann_kwds = {'index_type':index_type}
        adjacency_kwds = {'radius':radius, 'cyflann_kwds':cyflann_kwds}
        this_D = compute_adjacency_matrix(X=X, method = 'cyflann', **adjacency_kwds)
        assert_allclose(this_D.toarray(), D_true, rtol=1E-5, atol=1E-5)
    
    for index_type in ['kmeans', 'kdtrees']:
        yield check_index_type, index_type