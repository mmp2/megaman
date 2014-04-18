"""Riemannian Metric"""

# Author: Marina Meila <mmp@stat.washington.edu>
# License: BSD 3 clause

import warnings
import numpy as np

from scipy import sparse
from numpy.linalg import eigh, inv
#from scipy.numpy.dual import eigh, inv
#from scipy.sparse.linalg import lobpcg
#from scipy.sparse.linalg.eigen.lobpcg.lobpcg import symeig
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.externals import six
from sklearn.utils import check_random_state
#from sklearn.utils.validation import atleast2d_or_csr
#from sklearn.utils.graph import graph_laplacian
#from sklearn.utils.sparsetools import connected_components
#from sklearn.utils.arpack import eigsh
#from sklearn.metrics.pairwise import rbf_kernel
#from sklearn.neighbors import radius_neighbors_graph


d
def _set_diag(laplacian, value):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition

    Parameters
    ----------
    laplacian : array or sparse matrix
        The graph laplacian
    value : float
        The value of the diagonal

    Returns
    -------
    laplacian : array or sparse matrix
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        diag_idx = (laplacian.row == laplacian.col)
        laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


def riemann_meric( Y, n_dim=2, laplacian=None, adjacency=None, invert_h=False,
                   norm_laplacian=True, drop_first=True,
                   mode=None):
    """
    Parameters
    ----------
    adjacency : array-like or sparse matrix, shape: (n_samples, n_samples)
        The adjacency matrix of the graph to embed.

    n_dim : integer, optional
        The dimension of the projection subspace.

    Returns
    -------
    h_dual_metric : array, shape=(n_samples, n_dim, n_dim)

    Optionally:
    g_riemann_metric : array, shape=(n_samples, n_dim, n_dim)
    
    i would like to have a way to request g for specified points only

    Notes
    -----
    References
    ----------
    * 
    """
    n_nodes = adjacency.shape[0]
    # Whether to drop the first eigenvector
    if drop_first:
        n_dim = n_dim + 1
    # Check that either laplacian or symmetric adjacency matrix are given

    # If Laplacian not given, compute it from adjacency matrix.
    "to use sparse.csgraph.laplacian() and renormalize here?
    and put this calculation in the class anyways"
    if (laplacian == None ):
        laplacian, dd = graph_laplacian(adjacency,
                                        normed=norm_laplacian, return_diag=True)
    h_dual_metric = np.zeros((n_samples, n_dim, n_dim ))
    for ( i in 1:n_dim ):
        for( j in i:n_dim ):
            yij = Y[:,i]*Y[:,j]
            h_dual_metric[ :, i, j] = laplacian.dot(yij).todense()-laplacian.dot(Y[:,i]+Y[:,j]).todense()
            if ( j>i ):
                h_dual_metric[ :,j,i] = h_dual_metric[:,i,j]

    # compute rmetric if requested
    if( invert_h ):
        riemann_metric = np.zeros( n_samples, n_dim, n_dim )
        for( i in i 1:n_samples ):
            riemann_metric[i,:,:] = np.inv(h_dual_metric[i,:,:].squeeze())
    else:
        riemann_metric = None

    return h_dual_metric, riemann_metric, laplacian

""" MMP's plans: make a class RiemannMetric 
computes laplacian if needed
computes h on as many dim as given
computes g if requested with argument whichpoints of type index. if None compute on all points. optional argument compute g on a given number of dimensions.
other method e.g detect dimension

attributes h,g,n_dim max number dimensions,
 to also implement epsilon search => a different module?

"""
            

class RiemannMetric:
    """ 
    Parameters
    -----------
    Y: array, shape = (n_samples, n_dim )
    The embedding coordinates

    laplacian: sparse array, shape = (n_samples, n_samples )
    The Laplacian 
    For statistical correctness, the Laplacian should be computed using "radius_neighbors", weighted by RBF kernel, and "renormalized". The default laplacian in spectral_embedding satisfies these. No error is detected if these requirements are not satisfied.  
    
    optional:
    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.

    random_state : int seed, RandomState instance, or None, default : None
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg'.

    optional:
    affinity : string or callable, default : "radius_neighbors"
        How to construct the affinity matrix.
         - 'radius_neighbors' : construct affinity matrix by radius_neighbors_graph
         - 'rbf' : construct affinity matrix by rbf kernel
         - 'precomputed' : interpret X as precomputed affinity matrix
         - callable : use passed in function as affinity
           the function takes in data matrix (n_samples, n_features)
           and return affinity matrix (n_samples, n_samples).

    gamma : float, optional, default : 1/n_features
        Kernel coefficient for rbf kernel.


    Attributes
    ----------

    `embedding_` : array, shape = (n_samples, n_dim)
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
    MMP's notes: added radius_neighbors affinity with parameter neighbors_radius=None
    neighbors_radius = 1/sqrt(gamma)
    should we keep both params?
    to update the doc comments above
    """
    def __init__(self, n_dim=2, affinity="radius_neighbors",
                 gamma=None, random_state=None, eigen_solver=None,
                 neighbors_radius = None, n_neighbors=None):
        self.n_dim = n_dim
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.neighbors_radius = neighbors_radius
        self.n_neighbors = n_neighbors

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"

    def _get_affinity_matrix(self, X, Y=None):
        """Calculate the affinity matrix from data
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

            If affinity is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Returns
        -------
        affinity_matrix, shape (n_samples, n_samples)
        """
        if self.affinity == 'precomputed':
            self.affinity_matrix_ = X
            return self.affinity_matrix_
            
        # nearest_neigh kept for backward compatibility 
        if self.affinity == 'nearest_neighbors':
            if sparse.issparse(X):
                warnings.warn("Nearest neighbors affinity currently does "
                              "not support sparse input, falling back to "
                              "rbf affinity")
                self.affinity = "rbf"
            else:
                self.n_neighbors_ = (self.n_neighbors
                                     if self.n_neighbors is not None
                                     else max(int(X.shape[0] / 10), 1))
                self.affinity_matrix_ = kneighbors_graph(X, self.n_neighbors_)
                # currently only symmetric affinity_matrix supported
                self.affinity_matrix_ = 0.5 * (self.affinity_matrix_ +
                                               self.affinity_matrix_.T)
                return self.affinity_matrix_
        if self.affinity == 'radius_neighbors':
            self.neighbors_radius_ = (self.neighbors_radius
                           if self.neighbors_radius is not None else 1.0 / X.shape[1])   # to put another defaault value, like diam(X)/sqrt(dimensions)/10
            self.gamma_ = (self.gamma
                           if self.gamma is not None else 1.0 / X.shape[1])
            self.affinity_matrix_ = radius_neighbors_graph(X, self.neighbors_radius_, mode='distance')
            #problem here: this is a sparse matrix not np.array type
            #.todense() doesn't work -- i get a msg about a string

            self.affinity_matrix_ **= 2  # square distances
            self.affinity_matrix_ /= -self.neighbors_radius_**2  # less copying?
            self.affinity_matrix_ = np.exp( self.affinity_matrix_, self.affinity_matrix_ )
        if self.affinity == 'rbf':
            self.gamma_ = (self.gamma
                           if self.gamma is not None else 1.0 / X.shape[1])
            self.affinity_matrix_ = rbf_kernel(X, gamma=self.gamma_)
            return self.affinity_matrix_
        self.affinity_matrix_ = self.affinity(X)
        return self.affinity_matrix_

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

            If affinity is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        random_state = check_random_state(self.random_state)
        if isinstance(self.affinity, six.string_types):
            if self.affinity not in set(("nearest_neighbors", "rbf", 
                                         "radius_neighbors", "precomputed")):
                raise ValueError(("%s is not a valid affinity. Expected "
                                  "'precomputed', 'rbf', 'nearest_neighbors', 'radius_neighbors' "
                                  "or a callable.") % self.affinity)
        elif not callable(self.affinity):
            raise ValueError(("'affinity' is expected to be an an affinity "
                              "name or a callable. Got: %s") % self.affinity)

        affinity_matrix = self._get_affinity_matrix(X)
        self.embedding_ = spectral_embedding(affinity_matrix,
                                             n_dim=self.n_dim,
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
        X_new: array-like, shape (n_samples, n_dim)
        """
        self.fit(X)
        return self.embedding_
