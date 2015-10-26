"""Spectral Embedding"""

# Author: Marina Meila <mmp@stat.washington.edu>
#         James McQueen <jmcq@u.washington.edu>
#
#         after the scikit-learn version by 
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Wei LI <kuantkid@gmail.com>
# License: BSD 3 clause

import warnings
import numpy as np
import Mmani.embedding.geometry as geom
from Mmani.embedding.eigendecomp import eigen_decomposition

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.sparsetools import connected_components

def _is_symmetric(M, tol = 1e-8):
    if sparse.isspmatrix(M):
        conditions = np.abs((M - M.T).data) < tol 
    else:
        conditions = np.abs((M - M.T)) < tol
    return(np.all(conditions))

def _graph_connected_component(graph, node_id):
    """Find the largest graph connected components the contains one
    given node

    Parameters
    ----------
    graph : array-like, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes

    node_id : int
        The index of the query node of the graph

    Returns
    -------
    connected_components : array-like, shape: (n_samples,)
        An array of bool value indicates the indexes of the nodes
        belong to the largest connected components of the given query
        node
    """
    connected_components = np.zeros(shape=(graph.shape[0]), dtype=np.bool)
    connected_components[node_id] = True
    n_node = graph.shape[0]
    for i in range(n_node):
        last_num_component = connected_components.sum()
        _, node_to_add = np.where(graph[connected_components] != 0)
        connected_components[node_to_add] = True
        if last_num_component >= connected_components.sum():
            break
    return connected_components


def _graph_is_connected(graph):
    """ Return whether the graph is connected (True) or Not (False)

    Parameters
    ----------
    graph : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes

    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not
    """
    if sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]

def spectral_embedding(Geometry, n_components=8, eigen_solver=None,
                       random_state=None, eigen_tol=0.0, drop_first=True):
    """Project the sample on the first eigen vectors of the graph Laplacian.
    
    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigen vectors associated to the
    smallest eigen values) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.

    This embedding can also 'work' even if the ``adjacency`` variable is
    not strictly the adjacency matrix of a graph but more generally
    an affinity or similarity matrix between samples (for instance the
    heat kernel of a euclidean distance matrix or a k-NN matrix).

    However care must taken to always make the affinity matrix symmetric
    so that the eigen vector decomposition works as expected.

    Parameters
    ----------        
    Geometry : a Geometry object from Mmani.embedding.geometry

    n_components : integer, optional
        The dimension of the projection subspace.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.

    random_state : int seed, RandomState instance, or None (default)
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg'.
        By default, arpack is used.

    eigen_tol : float, optional, default=0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.

    drop_first : bool, optional, default=True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.

    Returns
    -------
    embedding : array, shape=(n_samples, n_components)
        The reduced samples.

    Notes
    -----
    Spectral embedding is most useful when the graph has one connected
    component. If there graph has many components, the first few eigenvectors
    will simply uncover the connected components of the graph.

    References
    ----------
    * http://en.wikipedia.org/wiki/LOBPCG

    * Toward the Optimal Preconditioned Eigensolver: Locally Optimal
      Block Preconditioned Conjugate Gradient Method
      Andrew V. Knyazev
      http://dx.doi.org/10.1137%2FS1064827500366124
    """

    random_state = check_random_state(random_state)    

    if not isinstance(Geometry, geom.Geometry):
        raise RuntimeError("Geometry object not Mmani.embedding.geometry Geometry class")
    affinity_matrix = Geometry.get_affinity_matrix()
    if not _graph_is_connected(affinity_matrix):
        warnings.warn("Graph is not fully connected, spectral embedding may not work as expected.")
    
    laplacian = Geometry.get_laplacian_matrix(return_lapsym = True, symmetrize = True)
    dd = laplacian.diagonal()
    lapl_type = Geometry.laplacian_type
    print lapl_type
    
    re_normalize = False
    if eigen_solver in ['amg', 'lobpcg']: # these methods require a symmetric matrix
        if lapl_type not in ['symmetricnormalized', 'unnormalized']:
            re_normalize = True
            # If a symmetric eigedecomposition algorithm is chosen and 
            # If the Laplacian is non-symmetric then we need to extract:
            # the w (weight) vector from geometry
            # and the symmetric Laplacian = S. 
            # The actual Laplacian is L = W^{-1}S  (Where W is the diagonal matrix of w)
            # Which has the same spectrum as: L* = W^{-1/2}SW^{-1/2} which is symmetric
            # We calculate the eigen-decomposition of L*: [D, V]
            # then use W^{-1/2}V  to compute the eigenvectors of L 
            # See (Handbook for Cluster Analysis Chapter 2 Proposition 1)
            w = np.array(Geometry.w)
            symmetrized_laplacian = Geometry.laplacian_symmetric.copy()
            assert(_is_symmetric(symmetrized_laplacian))
            if sparse.isspmatrix(symmetrized_laplacian):
                symmetrized_laplacian.data /= np.sqrt(w[symmetrized_laplacian.row])
                symmetrized_laplacian.data /= np.sqrt(w[symmetrized_laplacian.col])
            else:
                symmetrized_laplacian /= np.sqrt(w)
                symmetrized_laplacian /= np.sqrt(w[:,np.newaxis])
    if re_normalize:
        print 'using symmetrized laplacian'
        assert(_is_symmetric(symmetrized_laplacian))
        lambdas, diffusion_map = eigen_decomposition(symmetrized_laplacian, n_components, eigen_solver,
                                                    random_state, eigen_tol, drop_first)
        lambdas = 1 - lambdas
    else:
        lambdas, diffusion_map = eigen_decomposition(laplacian, n_components, eigen_solver,
                                                    random_state, eigen_tol, drop_first)
    if re_normalize:
        diffusion_map /= np.sqrt(w[:, np.newaxis]) # put back on original Laplacian space
        diffusion_map /= np.linalg.norm(diffusion_map, axis = 0) # norm 1 vectors
    # sort 
    ind = np.argsort(lambdas)
    ind = ind[::-1]
    lambdas = lambdas[ind]
    print lambdas
    diffusion_map = diffusion_map[:, ind]
    embedding = diffusion_map.T[n_components::-1]
    embedding *= dd
    if drop_first:
        embedding = embedding[1:n_components].T
    else:
        embedding = embedding[:n_components].T 
    return embedding


class SpectralEmbedding(BaseEstimator):
    """Spectral embedding for non-linear dimensionality reduction.

    Forms an affinity matrix given by the specified function and
    applies spectral decomposition to the corresponding graph laplacian.
    The resulting transformation is given by the value of the
    eigenvectors for each data point.

    Parameters
    -----------
    n_components : integer, default: 2
        The dimension of the projected subspace.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.

    random_state : int seed, RandomState instance, or None, default : None
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg'.

    is_affinity : string or callable, default : "nearest_neighbors"
         - True : interpret X as precomputed affinity matrix

    radius : float, optional, default : 1/n_features
        Kernel coefficient for rbf kernel.

    Attributes
    ----------

    `embedding_` : array, shape = (n_samples, n_components)
        Spectral embedding of the training matrix.

    `affinity_matrix_` : array, shape = (n_samples, n_samples)
        Affinity_matrix constructed from samples or precomputed.

    References
    ----------

    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    - On Spectral Clustering: Analysis and an algorithm, 2011
      Andrew Y. Ng, Michael I. Jordan, Yair Weiss
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.8100

    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324
    """

    """ 
    """
    def __init__(self, n_components=2, is_affinity = False,
                 radius=None, random_state=None, eigen_solver=None,
                 use_flann = False, path_to_flann = None, cpp_distances = False):
        self.n_components = n_components
        self.is_affinity = is_affinity
        self.radius = radius
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.use_flann = use_flann
        self.path_to_flann = path_to_flann
        self.cpp_distances = cpp_distances

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

            If is_affinity is True
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        random_state = check_random_state(self.random_state)
        if is_affinity:
            Geometry = geom.Geometry(X, neighbors_radius = self.radius, 
                                        is_affinity = True, use_flann = 
                                        self.use_flann, path_to_flann = 
                                        self.path_to_flann, cpp_distances = 
                                        self.cpp_distances)
        else:
            Geometry = geom.Geometry(X, neighbors_radius = self.radius, use_flann = 
                                        self.use_flann, path_to_flann = 
                                        self.path_to_flann, cpp_distances = 
                                        self.cpp_distances)
        self.embedding_ = spectral_embedding(Geometry,
                                             n_components=self.n_components,
                                             eigen_solver=self.eigen_solver,
                                             random_state=random_state)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

            If affinity is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.embedding_
