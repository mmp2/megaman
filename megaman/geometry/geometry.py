"""
Scalable Manifold learning utilities and algorithms.

Graphs are represented with their weighted adjacency matrices, preferably using
sparse matrices.

A note on symmetrization and internal sparse representations
------------------------------------------------------------

For performance, this code uses the FLANN library to compute
approximate neighborhoods efficiently. The down side of approximation
is that (1) the distance matrix (or adjacency matrix) produced is NOT
GUARANTEED to be symmetric. We also use sparse representations, and
(2) fl_radius_neighbors_graph returns a sparse matrix called distance_matrix.

distance_matrix has 0.0 on the diagonal, as it should. Implicitly, the
missing entries are infinity not 0 for this matrix. But (1) and (2)
mean that if one tries to symmetrize distance_matrix, the scipy.sparse
code eliminates the 0.0 entries from distance_matrix. In the Affinity
matrix we explicitly set the diagonal to 1.0 for sparse matrices.

Hence, I adopted the following convention:
   * distance_matrix will NOT BE GUARANTEED symmetric
   * affinity_matrix will perform a symmetrization by default
   * laplacian does NOT perform symmetrization by default, only if symmetrize=True, and DOES NOT check symmetry
   * these conventions are the same for dense matrices, for consistency
"""

# Authors: Marina Meila <mmp@stat.washington.edu>
#         James McQueen <jmcq@u.washington.edu>
# License: BSD 3 clause
from __future__ import division ## removes integer division
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist
import subprocess, os, sys, warnings

from .distance import distance_matrix
from ..utils.validation import check_array

sparse_formats = ['csr', 'coo', 'lil', 'bsr', 'dok', 'dia']
distance_methods = ['auto', 'brute', 'cyflann', 'pyflann', 'cython']
laplacian_types = ['symmetricnormalized', 'geometric', 'renormalized', 'unnormalized', 'randomwalk']

def symmetrize_sparse(A):
    """
    Symmetrizes a sparse matrix in place (coo and csr formats only)

    NOTES:
    1. if there are values of 0 or 0.0 in the sparse matrix, this operation will DELETE them.
    """
    if A.getformat() is not "csr":
        A = A.tocsr()
    A = (A + A.transpose(copy = True))/2
    return A

def affinity_matrix(distances, neighbors_radius, symmetrize = True):
    if neighbors_radius <= 0.:
        raise ValueError('neighbors_radius must be >0.')
    A = distances.copy()
    if sparse.isspmatrix( A ):
        A.data = A.data**2
        A.data = A.data/(-neighbors_radius**2)
        np.exp( A.data, A.data )
        if symmetrize:
            A = symmetrize_sparse( A )  # converts to CSR; deletes 0's
        else:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # sparse will complain that this is faster with lil_matrix
            A.setdiag(1) # the 0 on the diagonal is a true zero
    else:
        A **= 2
        A /= (-neighbors_radius**2)
        np.exp(A, A)
        if symmetrize:
            A = (A+A.T)/2
            A = np.asarray( A, order="C" )  # is this necessary??
        else:
            pass
    return A

###############################################################################
# Graph laplacian
# Code adapted from the Matlab function laplacian.m of Dominique Perrault-Joncas
def graph_laplacian(csgraph, normed = 'geometric', symmetrize = False,
                    scaling_epps = 0., renormalization_exponent = 1,
                    return_diag = False, return_lapsym = False):
    """
    Return the Laplacian matrix of an undirected graph.

    Computes a consistent estimate of the Laplace-Beltrami operator L
    from the similarity matrix A . See "Diffusion Maps" (Coifman and
    Lafon, 2006) and "Graph Laplacians and their Convergence on Random
    Neighborhood Graphs" (Hein, Audibert, Luxburg, 2007) for more
    details.

    A is the similarity matrix from the sampled data on the manifold M.
    Typically A is obtained from the data X by applying the heat kernel
    A_ij = exp(-||X_i-X_j||^2/EPPS). The bandwidth EPPS of the kernel is
    need to obtained the properly scaled version of L. Following the usual
    convention, the laplacian (Laplace-Beltrami operator) is defined as
    div(grad(f)) (that is the laplacian is taken to be negative
    semi-definite).

    Note that the Laplacians defined here are the negative of what is
    commonly used in the machine learning literature. This convention is used
    so that the Laplacians converge to the standard definition of the
    differential operator.

    notation: A = csgraph, D=diag(A1) the diagonal matrix of degrees
    L = lap = returned object, EPPS = scaling_epps**2

    Parameters
    ----------
    csgraph : array_like or sparse matrix, 2 dimensions
        compressed-sparse graph, with shape (N, N).
    normed : string, optional
        if 'renormalized':
            compute renormalized Laplacian of Coifman & Lafon
            L = D**-alpha A D**-alpha
            T = diag(L1)
            L = T**-1 L - eye()
        if 'symmetricnormalized':
           compute normalized Laplacian
            L = D**-0.5 A D**-0.5 - eye()
        if 'unnormalized': compute unnormalized Laplacian.
            L = A-D
        if 'randomwalks': compute stochastic transition matrix
            L = D**-1 A
    symmetrize: bool, optional
        if True symmetrize adjacency matrix (internally) before computing lap
    scaling_epps: float, optional
        if >0., it should be the same neighbors_radius that was used as kernel
        width for computing the affinity. The Laplacian gets the scaled by
        4/np.sqrt(scaling_epps) in order to ensure consistency in the limit
        of large N
    return_diag : bool, optional (kept for compatibility)
        If True, then return diagonal as well as laplacian.
    return_lapsym : bool, optional
        If normed in { 'geometric', 'renormalized' } then a symmetric matrix
        lapsym, and a row normalization vector w are also returned. Having
        these allows us to compute the laplacian spectral decomposition
        as a symmetric matrix, which has much better numerical properties.

    Returns
    -------
    lap : ndarray
        The N x N laplacian matrix of graph.
    diag : ndarray (obsolete, for compatibiility)
        The length-N diagonal of the laplacian matrix.
        diag is returned only if return_diag is True.

    Notes
    -----
    There are a few differences from the sklearn.spectral_embedding laplacian
    function.

    1. normed='unnormalized' and 'symmetricnormalized' correspond respectively
       to normed=False and True in the latter. (Note also that normed was changed
       from bool to string.
    2. the signs of this laplacians are changed w.r.t the original
    3. the diagonal of lap is no longer set to 0; also there is no checking if
       the matrix has zeros on the diagonal. If the degree of a node is 0, this
       is handled graciuously (by not dividing by 0).
    4. if csgraph is not symmetric the out-degree is used in the
       computation and no warning is raised. However, it is not recommended to
       use this function for directed graphs.
    """
    if csgraph.ndim != 2 or csgraph.shape[0] != csgraph.shape[1]:
        raise ValueError('csgraph must be a square matrix or array')

    normed = normed.lower()
    if normed not in ('unnormalized', 'geometric', 'randomwalk', 'symmetricnormalized','renormalized' ):
        raise ValueError('normed must be one of unnormalized, geometric, randomwalk, symmetricnormalized, renormalized')
    if (np.issubdtype(csgraph.dtype, np.int) or np.issubdtype(csgraph.dtype, np.uint)):
        csgraph = csgraph.astype(np.float)

    if sparse.isspmatrix(csgraph):
        return _laplacian_sparse(csgraph, normed = normed, symmetrize = symmetrize,
                                    scaling_epps = scaling_epps,
                                    renormalization_exponent = renormalization_exponent,
                                    return_diag = return_diag, return_lapsym = return_lapsym)

    else:
        return _laplacian_dense(csgraph, normed = normed, symmetrize = symmetrize,
                                scaling_epps = scaling_epps,
                                renormalization_exponent = renormalization_exponent,
                                return_diag = return_diag, return_lapsym = return_lapsym)

def _laplacian_sparse(csgraph, normed = 'geometric', symmetrize = True,
                        scaling_epps = 0., renormalization_exponent = 1,
                        return_diag = False, return_lapsym = False):
    n_nodes = csgraph.shape[0]
    lap = csgraph.copy()
    if symmetrize:
        if lap.format is not 'csr':
            lap.tocsr()
        lap = (lap + lap.T)/2.
    if lap.format is not 'coo':
        lap = lap.tocoo()
    diag_mask = (lap.row == lap.col)  # True/False
    degrees = np.asarray(lap.sum(axis=1)).squeeze()

    if normed == 'symmetricnormalized':
        w = np.sqrt(degrees)
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap.data /= w[lap.row]
        lap.data /= w[lap.col]
        lap.data[diag_mask] -= 1.
        if return_lapsym:
            lapsym = lap.copy()

    if normed == 'geometric':
        w = degrees.copy()     # normzlize one symmetrically by d
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap.data /= w[lap.row]
        lap.data /= w[lap.col]
        w = np.asarray(lap.sum(axis=1)).squeeze() #normalize again asymmetricall
        if return_lapsym:
            lapsym = lap.copy()
        lap.data /= w[lap.row]
        lap.data[diag_mask] -= 1.

    if normed == 'renormalized':
        w = degrees**renormalization_exponent;
        # same as 'geometric' from here on
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap.data /= w[lap.row]
        lap.data /= w[lap.col]
        w = np.asarray(lap.sum(axis=1)).squeeze() #normalize again asymmetricall
        if return_lapsym:
            lapsym = lap.copy()
        lap.data /= w[lap.row]
        lap.data[diag_mask] -= 1.

    if normed == 'unnormalized':
        lap.data[diag_mask] -= degrees
        if return_lapsym:
            lapsym = lap.copy()

    if normed == 'randomwalk':
        w = degrees.copy()
        if return_lapsym:
            lapsym = lap.copy()
        lap.data /= w[lap.row]
        lap.data[diag_mask] -= 1.
    if scaling_epps > 0.:
        lap.data *= 4/(scaling_epps**2)

    if return_diag:
        if return_lapsym:
            return lap, lap.data[diag_mask], lapsym, w
        else:
            return lap, lap.data[diag_mask]
    elif return_lapsym:
        return lap, lapsym, w
    else:
        return lap

def _laplacian_dense(csgraph, normed = 'geometric', symmetrize = True,
                        scaling_epps = 0., renormalization_exponent = 1,
                        return_diag = False, return_lapsym = False):
    n_nodes = csgraph.shape[0]
    if symmetrize:
        lap = (csgraph + csgraph.T)/2.
    else:
        lap = csgraph.copy()
    degrees = np.asarray(lap.sum(axis=1)).squeeze()
    di = np.diag_indices( lap.shape[0] )  # diagonal indices

    if normed == 'symmetricnormalized':
        w = np.sqrt(degrees)
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap /= w
        lap /= w[:, np.newaxis]
        di = np.diag_indices( lap.shape[0] )
        lap[di] -= (1 - w_zeros).astype(lap.dtype)
        if return_lapsym:
            lapsym = lap.copy()
    if normed == 'geometric':
        w = degrees.copy()     # normalize once symmetrically by d
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap /= w
        lap /= w[:, np.newaxis]
        w = np.asarray(lap.sum(axis=1)).squeeze() #normalize again asymmetricall
        if return_lapsym:
            lapsym = lap.copy()
        lap /= w[:, np.newaxis]
        lap[di] -= (1 - w_zeros).astype(lap.dtype)
    if normed == 'renormalized':
        w = degrees**renormalization_exponent;
        # same as 'geometric' from here on
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap /= w
        lap /= w[:, np.newaxis]
        w = np.asarray(lap.sum(axis=1)).squeeze() #normalize again asymmetricall
        if return_lapsym:
            lapsym = lap.copy()
        lap /= w[:, np.newaxis]
        lap[di] -= (1 - w_zeros).astype(lap.dtype)
    if normed == 'unnormalized':
        dum = lap[di]-degrees[np.newaxis,:]
        lap[di] = dum[0,:]
        if return_lapsym:
            lapsym = lap.copy()
    if normed == 'randomwalk':
        w = degrees.copy()
        if return_lapsym:
            lapsym = lap.copy()
        lap /= w[:,np.newaxis]
        lap -= np.eye(lap.shape[0])

    if scaling_epps > 0.:
        lap *= 4/(scaling_epps**2)

    if return_diag:
        diag = np.array( lap[di] )
        if return_lapsym:
            return lap, diag, lapsym, w
        else:
            return lap, diag
    elif return_lapsym:
        return lap, lapsym, w
    else:
        return lap

class Geometry:
    """
    The Geometry class stores the data, distance, affinity and laplacian
    matrices used by the various embedding methods and is the primary
    object passed to embedding functions.

    The Geometry class contains functions to build the aforementioned
    matrices and allows for re-computation whenever necessary.

    Parameters
    ----------
    X : array_like or sparse array. 2 dimensional. Value depends on input_type.
        size: (N_obs, N_dim) if 'data', (N_obs, N_obs) otherwise.
    input_type : string, one of: 'data', 'distance', 'affinity'. The values of X.
    neighborhood_radius : scalar, passed to distance_matrix. Value such that all
        distances beyond neighborhood_radius are considered infinite.
    affinity_radius : scalar, passed to affinity_matrix. 'bandwidth' parameter
        used in Guassian kernel for affinity matrix
    distance_method : string, one of 'auto', 'brute', 'cython', 'pyflann', 'cyflann'.
        method for computing pairwise radius neighbors graph.
    laplacian_type : string, one of: 'symmetricnormalized', 'geometric', 'renormalized',
        'unnormalized', 'randomwalk'
        type of laplacian to be computed. See graph_laplacian for more information.
    path_to_flann : string. full file path location of FLANN if not installed to root or
        FLANN_ROOT set to path location. Used for importing pyflann from a different location.

    """

    def __init__(self, X, neighborhood_radius = None, affinity_radius = None,
                 distance_method = 'auto', input_type = 'data',
                 laplacian_type = None, path_to_flann = None):
        self.distance_method = distance_method
        self.input_type = input_type
        self.path_to_flann = path_to_flann
        self.laplacian_type = laplacian_type

        if self.distance_method not in distance_methods:
            raise ValueError("invalid distance method.")

        if neighborhood_radius is None:
            self.neighborhood_radius = 1/X.shape[1]
        else:
            try:
                neighborhood_radius = np.float(neighborhood_radius)
                self.neighborhood_radius = neighborhood_radius
            except ValueError:
                raise ValueError("neighborhood_radius must be convertable to float")
        if affinity_radius is None:
            self.affinity_radius = self.neighborhood_radius
            self.default_affinity = True
        else:
            try:
                affinity_radius = np.float(affinity_radius)
                self.affinity_radius = affinity_radius
                self.default_affinity = False
            except ValueError:
                raise ValueError("affinity_radius must be convertable to float")

        if self.input_type == 'distance':
            X = check_array(X, accept_sparse = sparse_formats)
            a, b = X.shape
            if a != b:
                raise ValueError("input_type is distance but input matrix is not square")
            self.X = None
            self.distance_matrix = X
            self.affinity_matrix = None
            self.laplacian_matrix = None
        elif self.input_type == 'affinity':
            X = check_array(X, accept_sparse = sparse_formats)
            a, b = X.shape
            if a != b:
                raise ValueError("input_type is affinity but input matrix is not square")
            self.X = None
            self.distance_matrix = None
            self.affinity_matrix = X
            self.laplacian_matrix = None
        elif self.input_type == 'data':
            X = check_array(X, accept_sparse = sparse_formats)
            self.X = X
            self.distance_matrix = None
            self.affinity_matrix = None
            self.laplacian_matrix = None
        else:
            raise ValueError('input_type must be one of: data, distance, affinity.')

        if distance_method == 'cython':
            if input_type == 'data':
                try:
                    from .cyflann.index import Index
                except ImportError:
                    raise ValueError("distance_method set to cython but cyflann_index cannot be imported.")
                self.cyindex = Index(X)
        else:
            self.cyindex = None

        if distance_method == 'pyflann':
            if self.path_to_flann is not None:
                # FLANN is installed in specific location
                sys.path.insert(0, self.path_to_flann)
            try:
                import pyflann as pyf
            except ImportError:
                raise ValueError("distance_method is set to pyflann but pyflann is "
                                "not available.")
            self.flindex = pyf.FLANN()
            self.flparams = self.flindex.build_index(X, algorithm = 'kmeans',
                                                                         target_precision = 0.9)
        else:
            self.flindex = None
            self.flparams = None

    def get_distance_matrix(self, neighborhood_radius = None, copy = True):
        """
        Parameters
        ----------
        neighborhood_radius : scalar, passed to distance_matrix. Value such that all
            distances beyond neighborhood_radius are considered infinite.
            if this value is not passed the value of self.neighborhood_radius is used

        copy : boolean, whether to return a copied version of the distance matrix

        Returns
        -------
        self.distance_matrix : sparse Ndarray (N_obs, N_obs). Non explicit 0.0 values
            (e.g. diagonal) should be considered Infinite.
        """
        if self.input_type == 'affinity':
            raise ValueError("input_method was passed as affinity. "
                            "Distance matrix cannot be computed.")


        if self.distance_matrix is None:
            # if there's no existing distance matrix we make one
            if ((neighborhood_radius is not None) and (neighborhood_radius != self.neighborhood_radius)):
                # a different radius was passed than self.neighborhood_radius
                self.neighborhood_radius = neighborhood_radius
            self.distance_matrix = distance_matrix(self.X, method = self.distance_method,
                                                    flindex = self.flindex,
                                                    radius = self.neighborhood_radius,
                                                    cyindex = self.cyindex)
        else:
            # if there is an existing matrix we have to see if we need to overwrite
            if ((neighborhood_radius is not None) and (neighborhood_radius != self.neighborhood_radius)):
                # if there's a new radius we need to re-calculate
                if self.input_type == 'distance':
                    # but if we were passed distance this is impossible
                    raise ValueError("input_method was passed as distance."
                                    "requested radius not equal to self.neighborhood_radius."
                                    "distance matrix cannot be re-calculated.")
                else:
                    # if we were passed data then we need to re-calculate
                    self.neighborhood_radius = neighborhood_radius
                    self.distance_matrix = distance_matrix(self.X, method = self.distance_method,
                                                            flindex = self.flindex,
                                                            radius = self.neighborhood_radius,
                                                            cyindex = self.cyindex)

        if copy:
            return self.distance_matrix.copy()
        else:
            return self.distance_matrix

    def get_affinity_matrix(self, affinity_radius = None, copy = True,
                            symmetrize = True):
        """
        Parameters
        ----------
        affinity_radius : scalar, passed to affinity_matrix. 'bandwidth' parameter
            used in Guassian kernel for affinity matrix
            If this value is not passed then the self.affinity_radius value is used.

        copy : boolean, whether to return a copied version of the affinity matrix

        symmetrize : boolean, whether to explicitly symmetrize the affinity matrix.
            if distance_method = 'cython', 'cyflann', or 'pyflann' it is recommended
            to set this to True.

        Returns
        -------
        self.affinity_matrix : sparse Ndarray (N_obs, N_obs) contains the pairwise
            affinity values using the Guassian kernel and bandwidth equal to the
            affinity_radius
        """
        if self.affinity_matrix is None:
            # if there's no existing affinity matrix we make one
            if self.distance_matrix is None:
                # first check to see if we have the distance matrix
                self.distance_matrix = self.get_distance_matrix(copy = False)
            if affinity_radius is not None and affinity_radius != self.affinity_radius:
                self.affinity_radius = affinity_radius
                self.default_affinity = False
            self.affinity_matrix = affinity_matrix(self.distance_matrix,
                                                    self.affinity_radius, symmetrize)
        else:
            # if there is an existing matrix we have to see if we need to overrwite
            if (affinity_radius is not None and affinity_radius != self.affinity_radius) or (
                affinity_radius is not None and self.default_affinity):
                # if there's a new radius we need to re-calculate
                # or there's a passed radius and the current radius was set to default
                if self.input_type == 'affinity':
                    # but if we were passed affinity this is impossible
                    raise ValueError("Input_method was passed as affinity."
                                     "Requested radius was not equal to self.affinity_radius."
                                     "Affinity Matrix cannot be recalculated.")
                else:
                    # if we were passed distance or data we can recalculate:
                    if self.distance_matrix is None:
                        # first check to see if we have the distance matrix
                        self.distance_matrix = self.get_distance_matrix(copy = False)
                    self.affinity_radius = affinity_radius
                    self.default_affinity = False
                    self.affinity_matrix = affinity_matrix(self.distance_matrix,
                                                            self.affinity_radius, symmetrize)

        if copy:
            return self.affinity_matrix.copy()
        else:
            return self.affinity_matrix

    def get_laplacian_matrix(self, laplacian_type=None, symmetrize=False,
                            scaling_epps=None, renormalization_exponent=1,
                            copy=True, return_lapsym=False):
        """
        Parameters
        ----------
        laplacian_type : string, the type of graph laplacian to compute.
            see 'normed' in graph_laplacian for more information
        symmetrize : boolean, whether to pre-symmetrize the affinity matrix before
            computing the laplacian_matrix
        scaling_epps : scalar, the bandwidth/radius parameter used in the affinity matrix
            see graph_laplacian for more information
        renormalization_exponent : scalar, renormalization exponent for computing Laplacian
            see graph_laplacian for more information
        copy : boolean, whether to return copied version of the self.laplacian_matrix
        return_lapsym : boolean, if True returns additionally the symmetrized version of
            the requested laplacian and the re-normalization weights.

        Returns
        -------
        self.laplacian_matrix : sparse Ndarray (N_obs, N_obs). The requested laplacian.
        self.laplacian_symmetric : sparse Ndarray (N_obs, N_obs). The symmetric laplacian.
        self.w : Ndarray (N_obs). The renormalization weights used to make
            laplacian_matrix from laplacian_symmetric
        """
        # if scaling_epps is None:
            # scaling_epps = self.affinity_radius

        # First check if there's an existing Laplacian:
        if self.laplacian_matrix is not None:
            if (laplacian_type == self.laplacian_type) or (laplacian_type is None):
                if copy:
                    return self.laplacian_matrix.copy()
                else:
                    return self.laplacian_matrix
            else:
                warnings.warn("current Laplacian matrix is of type " + str(self.laplacian_type) +
                              " but type " + str(laplacian_type) + " was requested. "
                              "Existing Laplacian matrix will be overwritten.")

        # Next, either there is no Laplacian or we're replacing. Check type:
        if self.laplacian_type is None:
            if laplacian_type is None:
                laplacian_type = 'geometric' # default value
            self.laplacian_type = laplacian_type
        elif laplacian_type is not None and self.laplacian_type != laplacian_type:
            self.laplacian_type = laplacian_type

        # next check if we have an affinity matrix:
        if self.affinity_matrix is None:
            self.affinity_matrix = self.get_affinity_matrix(copy=False)

        # results depend on symmetric or not:
        if return_lapsym:
            (self.laplacian_matrix,
            self.laplacian_symmetric,
            self.w) = graph_laplacian(self.affinity_matrix, self.laplacian_type,
                                      symmetrize, scaling_epps, renormalization_exponent,
                                      return_diag=False, return_lapsym=True)
        else:
            self.laplacian_matrix = graph_laplacian(self.affinity_matrix,
                                                    self.laplacian_type,
                                                    symmetrize, scaling_epps,
                                                    renormalization_exponent)
        if copy:
            return self.laplacian_matrix.copy()
        else:
            return self.laplacian_matrix

    def assign_data_matrix(self, X):
        X = check_array(X, accept_sparse = sparse_formats)
        self.X = X

    def assign_distance_matrix(self, distance_mat, neighborhood_radius = None):
        distance_mat = check_array(distance_mat, accept_sparse = sparse_formats)
        (a, b) = distance_mat.shape
        if a != b:
            raise ValueError("distance matrix is not square")
        else:
            self.distance_matrix = distance_mat
            if neighborhood_radius is not None:
                self.neighborhood_radius = neighborhood_radius

    def assign_affinity_matrix(self, affinity_matrix, affinity_radius = None):
        affinity_matrix = check_array(affinity_matrix, accept_sparse = sparse_formats)
        (a, b) = affinity_matrix.shape
        if a != b:
            raise ValueError("affinity matrix is not square")
        else:
            self.affinity_matrix = affinity_matrix
            if affinity_radius is not None:
                self.affinity_radius = affinity_radius
                self.default_affinity = False

    def assign_laplacian_matrix(self, laplacian_matrix, laplacian_type = "unknown"):
        laplacian_matrix = check_array(laplacian_matrix, accept_sparse = sparse_formats)
        (a, b) = laplacian_matrix.shape
        if a != b:
            raise ValueError("Laplacian matrix is not square")
        else:
            self.laplacian_matrix = laplacian_matrix
            self.laplacian_type = laplacian_type;

    def assign_parameters(self, neighborhood_radius=None, affinity_radius=None,
                          distance_method=None, laplacian_type=None,
                          path_to_flann=None):
        """
        Note: self.neighborhood_radius, self.affinity_radius,
        and self.laplacian_type refer to the CURRENT
        version of these matrices.

        If you want to re-calculate with a new parameter DO NOT
        update these with assign_parameters, instead use
        get_distance_matrix(), get_affinity_matrix(), or get_laplacian_matrix()
        and pass the desired new parameter. This will automatically update
        the self.parameter version.

        If you change these values with assign_parameters Geometry will assume
        that the existing matrix follows that parameter and so, for example,
        calling get_distance_matrix() with a passed radius will *not*
        recalculate if the passed radius is equal to self.neighborhood_radius
        and there already exists a distance matrix.
        """
        if neighborhood_radius is not None:
            try:
                np.float(neighborhood_radius)
                self.neighborhood_radius = neighborhood_radius
            except ValueError:
                raise ValueError("neighborhood_radius must convertable to float")

        if affinity_radius is not None:
            try:
                np.float(affinity_radius)
                self.affinity_radius = affinity_radius
            except ValueError:
                raise ValueError("neighborhood_radius must convertable to float")

        if distance_method is not None:
            if distance_method in distance_methods:
                self.distance_method = distance_method
            else:
                raise ValueError("distance_method must be one of: ")

        if laplacian_type is not None:
            if laplacian_type in laplacian_types:
                self.laplacian_type = laplacian_type
            else:
                raise ValueError("laplacian_type method must be one of: ")

        if path_to_flann is not None:
            self.path_to_flann = path_to_flann
            sys.path.insert(0, self.path_to_flann)
            try:
                import pyflann as pyf
                self.flindex = pyf.FLANN()
                self.flparams = self.flindex.build_index(X, algorithm = 'kmeans',
                                                         target_precision = 0.9)
            except ImportError:
                raise ValueError("distance_method is set to pyflann but pyflann is "
                                "not available.")
