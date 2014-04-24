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

    Attributes
    ----------

    `embedding_` : array, shape = (n_samples, n_dim)
        Spectral embedding of the training matrix.

    `affinity_matrix_` : array, shape = (n_samples, n_samples)
        Affinity_matrix constructed from samples or precomputed.

    References
    ----------

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

