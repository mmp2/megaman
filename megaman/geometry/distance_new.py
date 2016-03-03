import numpy as np
from sklearn import neighbors
from scipy import sparse

from .cyflann.index import Index as CyIndex

try:
    import pyflann as pyf
    PYFLANN_LOADED = True
except ImportError:
    PYFLANN_LOADED = False

# from six.py
def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    return meta("NewBase", bases, {})


class AdjacencyMeta(type):
    """Metaclass for Adjacency object which registers subclasses"""
    def __init__(cls, name, bases, dct):
        if name == 'NewBase':
            # class created as part of six.with_metaclas
            pass
        elif not hasattr(cls, '_method_registry'):
            # this is the base class.  Create an empty registry
            cls._method_registry = {}
        else:
            # this is a derived class.  Add cls to the registry
            interface_id = name.lower()
            cls._method_registry[getattr(cls, 'name', name.lower())] = cls

        super(AdjacencyMeta, cls).__init__(name, bases, dct)


class Adjacency(with_metaclass(AdjacencyMeta)):
    """Base class for adjacency methods"""
    @classmethod
    def init(cls, method, *args, **kwargs):
        if method not in cls._method_registry:
            raise ValueError("method={0} not valid. Must be one of "
                             "{1}".format(method, list(cls.methods())))
        Adj = cls._method_registry[method]
        return Adj(*args, **kwargs)

    @classmethod
    def methods(cls):
        return cls._method_registry.keys()

    def __init__(self, radius=None, n_neighbors=None, mode='distance'):
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.mode = mode

        if (radius is None) == (n_neighbors is None):
           raise ValueError("Must specify either radius or n_neighbors, "
                            "but not both.")

    def adjacency_graph(self, X):
        if self.n_neighbors is not None:
            return self.knn_adjacency(X)
        elif self.radius is not None:
            return self.radius_adjacency(X)

    def knn_adjacency(self, X):
        raise NotImplementedError()

    def radius_adjacency(self, X):
        raise NotImplementedError()


class BruteForceAdjacency(Adjacency):
    name = 'brute'

    def radius_adjacency(self, X):
        model = neighbors.NearestNeighbors(algorithm=self.name).fit(X)
        return model.radius_neighbors_graph(radius=self.radius,
                                            mode=self.mode)

    def knn_adjacency(self, X):
        model = neighbors.NearestNeighbors(algorithm=self.name).fit(X)
        return model.kneighbors_graph(n_neighbors=self.n_neighbors,
                                      mode=self.mode)


class KDTreeAdjacency(BruteForceAdjacency):
    name = 'kd_tree'


class BallTreeAdjacency(BruteForceAdjacency):
    name = 'ball_tree'


class CyFLANNAdjacency(Adjacency):
    name = 'cyflann'

    def __init__(self, radius=None, n_neighbors=None, flann_index=None,
                 target_precision=0.9, cyflann_kwds=None):
        self.flann_index = flann_index
        self.target_precision = target_precision
        self.cyflann_kwds = cyflann_kwds
        super(CyFLANNAdjacency, self).__init__(radius=radius,
                                               n_neighbors=n_neighbors,
                                               mode='distance')

    def _get_built_index(self, X):
        if self.flann_index is None:
            cyindex = CyIndex(X, target_precision=self.target_precision,
                              **(self.cyflann_kwds or {}))
        else:
            cyindex = self.flann_index
        cyindex.buildIndex()
        return cyindex

    def radius_adjacency(self, X):
        cyindex = self._get_built_index(X)
        return cyindex.radius_neighbors_graph(X, self.radius)

    def knn_adjacency(self, X):
        cyindex = self._get_built_index(X)
        return cyindex.knn_neighbors_graph(X, self.n_neighbors)


class PyFLANNAdjacency(Adjacency):
    name = 'pyflann'

    def __init__(self, radius=None, n_neighbors=None, flann_index=None,
                 algorithm='kmeans', target_precision=0.9, pyflann_kwds=None):
        if not PYFLANN_LOADED:
            raise ValueError("pyflann must be installed "
                             "to use method='pyflann'")
        self.flann_index = flann_index
        self.algorithm = algorithm
        self.target_precision = target_precision
        self.pyflann_kwds = pyflann_kwds
        super(PyFLANNAdjacency, self).__init__(radius=radius,
                                               n_neighbors=n_neighbors,
                                               mode='distance')

    def _get_built_index(self, X):
        if self.flann_index is None:
            pyindex = pyf.FLANN(**(self.pyflann_kwds or {}))
        else:
            pyindex = self.flann_index

        flparams = pyindex.build_index(X, algorithm=self.algorithm,
                                       target_precision=self.target_precision)
        return pyindex

    def radius_adjacency(self, X):
        flindex = self._get_built_index(X)

        n_samples, n_features = X.shape
        X = np.require(X, requirements = ['A', 'C']) # required for FLANN

        graph_i = []
        graph_j = []
        graph_data = []
        for i in range(n_samples):
            jj, dd = flindex.nn_radius(X[i], self.radius ** 2)
            graph_data.append(dd)
            graph_j.append(jj)
            graph_i.append(i*np.ones(jj.shape, dtype=int))

        graph_data = np.concatenate(graph_data)
        graph_i = np.concatenate(graph_i)
        graph_j = np.concatenate(graph_j)
        return sparse.coo_matrix((np.sqrt(graph_data), (graph_i, graph_j)),
                                 shape=(n_samples, n_samples))

    def knn_adjacency(self, X):
        n_samples = X.shape[0]
        flindex = self._get_built_index(X)
        A_ind, A_data = flindex.nn_index(X, self.n_neighbors)
        A_ind = np.ravel(A_ind)
        A_data = np.sqrt(np.ravel(A_data))  # FLANN returns square distances
        A_indptr = self.n_neighbors * np.arange(n_samples + 1)
        return sparse.csr_matrix((A_data, A_ind, A_indptr),
                                 shape=(n_samples, n_samples))


def adjacency_graph(X, method='auto', **kwargs):
    """Compute an adjacency graph with the given method"""
    if method == 'auto':
        if X.shape[0] > 10000:
            method = 'cyflann'
        else:
            method = 'kd_tree'
    return Adjacency.init(method, **kwargs).adjacency_graph(X)
