"""Embedding by Isomap"""
# adapted from spectral_embedding_.py in progress

# Author: Marina Meila <mmp@stat.washington.edu>
#
#         after the scikit-learn version by 
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Wei LI <kuantkid@gmail.com>
# License: BSD 3 clause

import warnings
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import lobpcg
from scipy.sparse.linalg.eigen.lobpcg.lobpcg import symeig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
from sklearn.utils import check_random_state
from sklearn.utils.validation import atleast2d_or_csr
from sklearn.utils.graph import graph_laplacian
from sklearn.utils.sparsetools import connected_components
from sklearn.utils.arpack import eigsh
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import kneighbors_graph
timing = False


def isomap_embedding(distance_matrix, n_components=2, mode=None, 
                     path_method=Noneeigen_solver=None,
                     tol=0, max_iter=None )
    """
    MMP:TO update THIS

    Parameters
    ----------
    distance_matrix : array-like or sparse matrix, shape: (n_samples, n_samples)
        The adjacency matrix of the graph to embed.

    n_components : integer, optional
        The dimension of the projection subspace.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.

        By default, arpack is used.

    tol : float, optional, default=0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.

    Returns
    -------
    embedding : array, shape=(n_samples, n_components)
        The reduced samples.

    Notes
    -----


    References
    ----------
    * http://en.wikipedia.org/wiki/LOBPCG

    * Toward the Optimal Preconditioned Eigensolver: Locally Optimal
      Block Preconditioned Conjugate Gradient Method
      Andrew V. Knyazev
      http://dx.doi.org/10.1137%2FS1064827500366124
    """

    # Check that the matrice given is symmetric
    all_dist = graph_shortest_path( distance_matrix, method=path_method,
                                    directed=False)
    # this is dense make it sparse too
    G = all_dist** 2
    G *= -0.5

    if mode="kernelPCA":
        kpca = KernelPCA(n_components, kernel="precomputed",
                         eigen_solver=eigen_solver, tol=tol, max_iter=max_iter)

        embedding = kpca.fit_transform(G)
        return embedding, all_dist, kpca
    else if mode="smacof":
        embedding, stress = smacof(all_dist, metric=True, n_components=n_components,
                                   init=None, n_init=8, n_jobs=1, max_iter=300,
                                   verbose=0, eps=1e-3, random_state=None)
        #do i need to pass in more params? do i need to initialize/create
        return embedding, all_dist, stress


class Isomap(BaseEstimator, TransformerMixin):
    """Spectral embedding for non-linear dimensionality reduction.

    Forms an affinity matrix given by the specified function and
    applies spectral decomposition to the corresponding graph laplacian.
    The resulting transformation is given by the value of the
    eigenvectors for each data point.

    Parameters
    -----------
    n_components : integer, default: 2
        The dimension of the projected subspace.
    gamma : float, optional, default : 1/n_features
        Kernel coefficient for rbf kernel.

    n_neighbors : int, default : max(n_samples/10 , 1)
        Number of nearest neighbors for nearest_neighbors graph building.

    Attributes
    ----------

    `embedding_` : array, shape = (n_samples, n_components)
        Spectral embedding of the training matrix.
    References
    ----------
    """

    """ 
    MMP's notes: added radius_neighbors affinity with parameter neighbors_radius=None
    neighbors_radius = 1/sqrt(gamma)
    should we keep both params?
    to update the doc comments above
    """
    def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto',
                 tol=0, max_iter=None, path_method='auto',
                 neighbors_algorithm="radius_neighbors", gamma=None,
                 neighbors_radius=None ):

        self.distance = distance
        self.n_components = n_components
        self.gamma = gamma
        self.eigen_solver = eigen_solver
        self.neighbors_radius = neighbors_radius
        self.n_neighbors = n_neighbors
        self.tol_ = tol
        self.max_iter = max_iter
        self.path_method = path_method

    @property
    def _pairwise(self):
        return self.distance == "precomputed"

    def _get_distance_matrix(self, X, Y=None):
        """Calculate the distance matrix from data
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

            If distance is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Returns
        -------
        distance_matrix, shape (n_samples, n_samples)
        """
        if self.distance == 'precomputed':
            self.distance_matrix_ = X
            print( type( self.distance_matrix_))
            return self.distance_matrix_
            
        # nearest_neigh kept for backward compatibility 
        if self.distance == 'nearest_neighbors':
            if sparse.issparse(X):
                warnings.warn("Nearest neighbors distance currently does "
                              "not support sparse input, falling back to "
                              "rbf distance")
                self.distance = "rbf"
            else:
                self.n_neighbors_ = (self.n_neighbors
                                     if self.n_neighbors is not None
                                     else max(int(X.shape[0] / 10), 1))
                self.distance_matrix_ = kneighbors_graph(X, self.n_neighbors_)
                # currently only symmetric distance_matrix supported
                self.distance_matrix_ = 0.5 * (self.distance_matrix_ +
                                               self.distance_matrix_.T)
                return self.distance_matrix_
        if self.distance == 'radius_neighbors':
            if self.neighbors_radius is None:
                self.neighbors_radius_ =  np.sqrt(X.shape[1])
                # to put another default value, like diam(X)/sqrt(dimensions)/10
            else:
                self.neighbors_radius_ = self.neighbors_radius
                
            self.gamma_ = (self.gamma
                           if self.gamma is not None else 1.0 / X.shape[1])
            self.distance_matrix_ = radius_neighbors_graph(X, self.neighbors_radius_, mode='distance')
            return self.distance_matrix_

    def get_distance_matrix(self, X=None, Y=None, copy = True):
        if self.distance_matrix_ == None:
            if X is None: 
                return None
            _get_distance_matrix(self, X, Y)
        if copy:
            return self.distance_matrix_.copy()
        else:
            return self.distance_matrix_

# the matrices are very large and  i don;'t really want to clone. i'll chabnge later

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

            If distance is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if isinstance(self.distance, six.string_types):
            if self.distance not in set(("nearest_neighbors", 
                                         "radius_neighbors", "precomputed")):
                raise ValueError(("%s is not a valid distance. Expected "
                                  "'precomputed', 'rbf', 'nearest_neighbors', 'radius_neighbors' "
                                  "or a callable.") % self.distance)
        elif not callable(self.distance):
            raise ValueError(("'distance' is expected to be an an distance "
                              "name or a callable. Got: %s") % self.distance)
        if timing:
            import time
            t0 = time.time()
        distance_matrix = self._get_distance_matrix(X)
        print( type( distance_matrix ))
        if timing:
            ta = time.time()
        self.embedding_, self.all_dist_ = isomap_embedding(distance_matrix,
                                           self.tol_, self.max_iter_, self.path_method,                                             
                                           n_components=self.n_components,
                                           eigen_solver=self.eigen_solver)
        if timing:
            tl = time.time()
            print('timing: distance = {0}s, embedding = {1}s'.format( ta-t0, tl-ta ))
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

            If distance is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.embedding_


""" written isomap_embedding(), __init__. fit() etc are from spectral embeding.
to see what other functions are in isomap-copy.py
 to make parameters agree overall
small functionality like access to class members.
tests
sparse matrix implementation--- does it make sense??
"""
