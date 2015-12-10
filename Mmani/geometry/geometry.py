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
code eliminates the 0.0 entries from distance_matrix. [I did not find
an efficient way around this problem.]

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
from Mmani.geometry.distance import distance_matrix

def symmetrize_sparse(A):
    """
    Symmetrizes a sparse matrix in place (coo and csr formats only)

    NOTES: 
      1) if there are values of 0 or 0.0 in the sparse matrix, this operation will DELETE them. 
    """
    if A.getformat() is not "csr":
        A = A.tocsr()
    A = (A + A.transpose(copy = True))/2
    
def affinity_matrix(distances, neighbors_radius, symmetrize = True):
    if neighbors_radius <= 0.:
        raise ValueError('neighbors_radius must be >0.')
    A = distances.copy()
    if sparse.isspmatrix( A ):
        A.data = A.data**2
        A.data = A.data/(-neighbors_radius**2)
        np.exp( A.data, A.data )
        if symmetrize:
            symmetrize_sparse( A )  # converts to CSR; deletes 0's
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
    """ Return the Laplacian matrix of an undirected graph.

   Computes a consistent estimate of the Laplace-Beltrami operator L
   from the similarity matrix A . See "Diffusion Maps" (Coifman and
   Lafon, 2006) and "Graph Laplacians and their Convergence on Random
   Neighborhood Graphs" (Hein, Audibert, Luxburg, 2007) for more
   details. 

   ????It also returns the Kth firts eigenvectors PHI of the L in
   increasing order of eigenvalues LAM.

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

    Parameters
    ----------
    notation: A = csgraph, D=diag(A1) the diagonal matrix of degrees
              L = lap = returned object
              EPPS = scaling_epps**2
           
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
    1) normed='unnormalized' and 'symmetricnormalized' correspond 
    respectively to normed=False and True in the latter. (Note also that normed
    was changed from bool to string.
    2) the signs of this laplacians are changed w.r.t the original
    3) the diagonal of lap is no longer set to 0; also there is no checking if 
    the matrix has zeros on the diagonal. If the degree of a node is 0, this
    is handled graciuously (by not dividing by 0).
    4) if csgraph is not symmetric the out-degree is used in the
    computation and no warning is raised. 
    However, it is not recommended to use this function for directed graphs.
    Use directed_laplacian() (NYImplemented) instead
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
    def __init__(self, X, neighborhood_radius = None, affinity_radius = None,
                 distance_method = 'auto', input_type = 'data', 
                 laplacian_type = None, path_to_pyflann = None):
        """ parameters:
            X 
            neighborhood_radius 
            affinity_radius
            laplacian_type
            path_to_pyflann
            distance_method: 'auto', 'pyflann', 'cyflann', 'brute'
            input_type: 'data', 'distance', 'affinity'
        """
        self.distance_method = distance_method
        self.input_type = input_type
        self.path_to_pyflann = path_to_pyflann
        self.laplacian_type = laplacian_type
        
        if neighborhood_radius is None:
            self.neighborhood_radius = 1/X.shape[1]
        else:
            self.neighborhood_radius = neighborhood_radius
        if affinity_radius is None:
            self.affinity_radius = self.neighborhood_radius
        else:
            self.affinity_radius = affinity_radius
        
        if self.input_type == 'distance':
            a, b = X.shape
            if a != b:
                raise ValueError("input_type is distance but input matrix is not square")
            self.X = None
            self.distance_matrix = X
        elif self.input_type == 'affinity':
            a, b = X.shape
            if a != b:
                raise ValueError("input_type is affinity but input matrix is not square")
            self.X = None
            self.distance_matrix = None
            self.affinity_matrix = X
        else:
            self.X = X
            self.distance_matrix = None
            self.affinity_matrix = None
        
        if distance_method == 'pyflann':
            if self.path_to_flann is not None: 
                # FLANN is installed in specific location
                sys.path.insert(0, self.path_to_flann)
            try:
                import pyflann as pyf
                self.flindex = pyf.FLANN()
                self.flparams = self.flindex.build_index(X, algorithm = 'kmeans', 
                                                         target_precision = 0.9)
            except ImportError:
                raise ValueError("distance_method is set to pyflann but pyflann is "
                                "not available.")
        else:
            self.flindex = None
            self.flparams = None
        
    def get_distance_matrix(self, neighborhood_radius = None, copy = True):
        if neighborhood_radius is None:
            radius = self.neighborhood_radius
        else:
            radius = neighborhood_radius
        if self.input_type == 'affinity':
            raise ValueError("input_method was passed as affinity. "
                            "Distance matrix cannot be computed.")
        elif self.distance_matrix is None:
            self.distance_matrix = distance_matrix(self.X, method = self.distance_method,
                                                    flindex = self.flindex, 
                                                    radius = self.neighborhood_radius)
        if copy:
            return self.distance_matrix.copy()
        else:
            return self.distance_matrix
    
    def get_affinity_matrix(self, affinity_radius = None, copy = True, 
                            symmetrize = True):
        if affinity_radius is None:
            radius = self.affinity_radius
        else:
            radius = affinity_radius
        if self.affinity_matrix is None:
            if self.distance_matrix is None:
                self.distance_matrix = self.get_distance_matrix(copy = False)
            self.affinity_matrix = affinity_matrix(self.distance_matrix, 
                                                    radius, symmetrize)
        if copy:
            return self.affinity_matrix.copy()
        else:
            return self.affinity_matrix
                
    def get_laplacian_matrix(self, normed=None, symmetrize=True,
                            scaling_epps=0., renormalization_exponent=1,
                            copy=True, return_lapsym=False):
        # First check if there's an existing Laplacian: 
        if hasattr(self, 'laplacian_matrix'):
            if (normed == self.laplacian_type) or (normed is None):
                if copy:
                    return self.laplacian_matrix.copy()
                else:
                    return self.laplacian_matrix
            else:
                warnings.warn("current Laplacian matrix is of type " + str(self.laplacian_type) + 
                              " but type " + str(normed) + " was requested."
                              "Existing Laplacian matrix will be overwritten.")
                              
        # Next, either there is no Laplacian or we're replacing. Check type:
        if self.laplacian_type is None:
            if normed is None:
                normed = 'geometric' # default value
            self.laplacian_type = normed
        elif normed is not None and self.laplacian_type != normed:
            self.laplacian_type = normed
            
        # next check if we have a distance matrix:
        if self.affinity_matrix is None:
            self.affinity_matrix = self.get_affinity_matrix()
            
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
    
    def assign_distance_matrix(self, distance_matrix):
        (a, b) = distance_matrix.shape
        if a != b:
            raise ValueError("distance matrix is not square")
        else:
            self.distance_matrix = distance_matrix
    
    def assign_affinity_matrix(self, affinity_matrix):
        (a, b) = affinity_matrix.shape
        if a != b:
            raise ValueError("affinity matrix is not square")
        else:
            self.affinity_matrix = affinity_matrix
    
    def assign_laplacian_matrix(self, laplacian_matrix, normed = "unknown"):
        (a, b) = laplacian_matrix.shape
        if a != b:
            raise ValueError("Laplacian matrix is not square")
        else:
            self.laplacian_matrix = laplacian_matrix
            self.laplacian_type = normed;
    
    def assign_neighbors_radius(self, radius):
        self.neighbors_radius = radius
        
    def assign_affinity_radius(self, radius):
        self.affinity_radius = radius
