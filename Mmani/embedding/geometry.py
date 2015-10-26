"""
Scalable Manifold learning utilities and algorithms. 

Graphs are represented with their weighted adjacency matrices, preferably using
sparse matrices.

A note on symmetrization and internal sparse representations
------------------------------------------------------------ 

For performance, this code uses the FLANN libarary to compute
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
#Authors: Marina Meila <mmp@stat.washington.edu>
#         James McQueen <jmcq@u.washington.edu>
# License: BSD 3 clause
from __future__ import division ## removes integer division
import numpy as np
from scipy import sparse
from sklearn.neighbors import radius_neighbors_graph
import subprocess, os, sys

def _is_symmetric(M, tol = 1e-8):
    if sparse.isspmatrix(M):
        conditions = np.abs((M - M.T).data) < tol 
    else:
        conditions = np.abs((M - M.T)) < tol
    return(np.all(conditions))

def distance_matrix(X, flindex = None, neighbors_radius = None, cpp_distances = False):
        if neighbors_radius is None:
            neighbors_radius = 1/X.shape[1]
        if flindex is not None:
            distance_matrix = fl_radius_neighbors_graph(X, neighbors_radius, flindex, mode='distance')
        elif cpp_distances:
            distance_matrix = fl_cpp_radius_neighbors_graph(X, neighbors_radius)
        else:
            distance_matrix = radius_neighbors_graph(X, neighbors_radius, mode='distance')
        return distance_matrix
        
def fl_cpp_radius_neighbors_graph(X, radius):
    """
    Constructs a sparse distance matrix called graph in coo
    format. 
    Parameters
    ----------
    X: data matrix, array_like, shape = (n_samples, n_dimensions )
    radius: neighborhood radius, scalar
        the neighbors lying approximately within radius of a node will
        be returned. Or, in other words, all distances will be less or equal
        to radius. There will be entries in the matrix for zero distances.
        
        Attention when converting to dense: The rest of the distances
        should not be considered 0, but "large".
    
    Returns
    -------
    graph: the distance matrix, array_like, shape = (X.shape[0],X.shape[0])
           sparse csr format
    
    Notes
    -----
    With approximate neiborhood search, the matrix is not necessarily symmetric. 
    """
    # To run the C++ executable we must save the data to a binary file
    nsam, ndim = X.shape
    fname = os.getcwd() + "/test_data_file.dat"
    X.tofile(fname)
    geom_path = os.path.abspath(__file__) # this is where geometry.py is located
    split_path = geom_path.split("/")
    split_path[-1] = "compute_flann_neighbors_cpp"
    cpp_file_path = "/".join(split_path)
    unix_call = "{file_path} {N} {D} {radius} {fname}"
    dist_call = unix_call.format(file_path = cpp_file_path, N=nsam, D=ndim, 
                                    radius=radius, fname=fname)     
    print dist_call 
    ret_code = subprocess.call([dist_call], shell = True)
    if ret_code != 0:
        raise RuntimeError("shell call: " + dist_call + " failed with code: " + str(ret_code))
    
    # the resulting files from the C++ function are:
    # neighbors.dat: contains nsam rows with space separated index of nbr
    # distances.dat: contains nsam rows with space separated distance of nbr
    with open("neighbors.dat", "rb") as handle:
        allnbrs = handle.readlines()
    with open("distances.dat", "rb") as handle:
        alldists = handle.readlines()
    allnbrs = [nbrs.split() for nbrs in allnbrs]
    alldists = [dists.split() for dists in alldists]
    
    # for sparse storage
    indices = np.array([int(nbr) for nbri in allnbrs for nbr in nbri])
    data = np.array([float(dist) for disti in alldists for dist in disti])
    lengths = [len(nbri) for nbri in allnbrs]
    indpts = list(np.cumsum(lengths))
    indpts.insert(0,0)
    indpts = np.array(indpts)
    graph = sparse.csr_matrix((data, indices, indpts), shape = (nsam, nsam))
    return graph
    
def fl_radius_neighbors_graph(X, radius, flindex):
    """
    Constructs a sparse distance matrix called graph in coo
    format. 
    Parameters
    ----------
    X: data matrix, array_like, shape = (n_samples, n_dimensions )
    radius: neighborhood radius, scalar
        the neighbors lying approximately within radius of a node will
        be returned. Or, in other words, all distances will be less or equal
        to radius. There will be entries in the matrix for zero distances.
        
        Attention when converting to dense: The rest of the distances
        should not be considered 0, but "large".
   
    flindex: FLANN index of the data X

    Returns
    -------
    graph: the distance matrix, array_like, shape = (X.shape[0],X.shape[0])
           sparse coo or csr format
    
    Notes
    -----
    With approximate neighborhood search, the matrix is not
    necessarily symmetric. 

    mode = 'adjacency' not implemented
    """
    if radius < 0.:
        raise ValueError('neighbors_radius must be >=0.')
    nsam, ndim = X.shape
    X = np.require(X, requirements = ['A', 'C']) # required for FLANN
    
    graph_jindices = []
    graph_iindices = []
    graph_data = []
    for i in xrange(nsam):
        jj, dd = flindex.nn_radius(X[i,:], radius)
        graph_data.append(dd)
        graph_jindices.append(jj)
        graph_iindices.append(i*np.ones(jj.shape, dtype=int))

    graph_data = np.concatenate( graph_data )
    graph_iindices = np.concatenate( graph_iindices )
    graph_jindices = np.concatenate( graph_jindices )
    graph = sparse.coo_matrix((graph_data, (graph_iindices, graph_jindices)), shape=(nsam, nsam))
    return graph

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
                                return_diag = return_diag)

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
        # same as 'geoetric' from here on
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
    if normed == 'randomwalk':
        w = degres.copy()
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

def single_source_shortest_path_length(graph, source, cutoff=None):
    """Return the shortest path length from source to all reachable nodes.

    Returns a dictionary of shortest path lengths keyed by target.

    Parameters
    ----------
    graph: sparse matrix or 2D array (preferably LIL matrix)
        Adjacency matrix of the graph
    source : node label
       Starting node for path
    cutoff : integer, optional
        Depth to stop the search - only
        paths of length <= cutoff are returned.

    Examples
    --------
    >>> import numpy as np
    >>> graph = np.array([[ 0, 1, 0, 0],
    ...                   [ 1, 0, 1, 0],
    ...                   [ 0, 1, 0, 1],
    ...                   [ 0, 0, 1, 0]])
    >>> single_source_shortest_path_length(graph, 0)
    {0: 0, 1: 1, 2: 2, 3: 3}
    >>> single_source_shortest_path_length(np.ones((6, 6)), 2)
    {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 1}
    """
    if sparse.isspmatrix(graph):
        graph = graph.tolil()
    else:
        graph = sparse.lil_matrix(graph)
    seen = {}                   # level (number of hops) when seen in BFS
    level = 0                   # the current level
    next_level = [source]       # dict of nodes to check at next level
    while next_level:
        this_level = next_level     # advance to next level
        next_level = set()          # and start a new list (fringe)
        for v in this_level:
            if v not in seen:
                seen[v] = level     # set the level of vertex v
                next_level.update(graph.rows[v])
        if cutoff is not None and cutoff <= level:
            break
        level += 1
    return seen  # return all path lengths as dictionary
    
class Geometry:
    """ The main class of this package. A Geometry object will contain all 
    of the geometric information regarding the original data set. 
    
    All embedding functions either accept a Geometry object or create one. 
    """
    def __init__(self, X, use_flann = False, neighbors_radius = None, 
                is_distance = False, is_affinity = False,
                cpp_distances = False, path_to_flann = None):
        self.neighbors_radius = neighbors_radius
        self.is_distance = is_distance
        self.is_affinity = is_affinity
        self.cpp_distances = cpp_distances
        self.path_to_flann = path_to_flann
        if self.is_distance and self.is_affinity:
            raise ValueError("cannot be both distance and affinity")
        if self.is_distance:
            a, b = X.shape
            if a != b:
                raise ValueError("is_distance is True but X not square")
            self.X = None
            self.distance_matrix = X
        elif self.is_affinity:
            a, b = X.shape
            if a != b:
                raise ValueError("is_affinity is True but X not square")
            self.X = None
            self.distance_matrix = None
            self.affinity_matrix = X
        else:
            self.X = X
            self.distance_matrix = None
            self.affinity_matrix = None
        if use_flann:
            if self.path_to_flann is not None: # FLANN is installed in specific location
                sys.path.insert(0, self.path_to_flann)
            try:
                import pyflann as pyf
                self.flindex = pyf.FLANN()
                self.flparams = self.flindex.build_index(X, algorithm = 'kmeans', target_precision = 0.9)
            except ImportError:
                raise ValueError("use_flann is set to True but pyflann is "
                                "not available.")
        else:
            self.flindex = None
            self.flparams = None
    # Functions to get distance, affinity, and Laplacian matrices:
    def get_distance_matrix(self, neighbors_radius = None, copy = True):
        if neighbors_radius is None:
            radius = self.neighbors_radius
        else:
            radius = neighbors_radius
        if self.is_affinity:
            raise ValueError("is_affinity was passed as true. "
                            "Distance matrix cannot be computed.")
        elif self.distance_matrix is None:
            self.distance_matrix = distance_matrix(self.X, self.flindex, radius, 
                                                    self.cpp_distances)
        if copy:
            return self.distance_matrix.copy()
        else:
            return self.distance_matrix
    
    def get_affinity_matrix(self, neighbors_radius = None, copy = True, 
                            symmetrize = True):
        if neighbors_radius is None:
            radius = self.neighbors_radius
        else:
            radius = neighbors_radius
        if self.affinity_matrix is None:
            if self.distance_matrix is None:
                self.distance_matrix = self.get_distance_matrix(copy = False)
            self.affinity_matrix = affinity_matrix(self.distance_matrix, 
                                                    radius, symmetrize)
        if copy:
            return self.affinity_matrix.copy()
        else:
            return self.affinity_matrix
    
    def get_laplacian_matrix(self, normed='geometric', symmetrize=True, 
                            scaling_epps=0., renormalization_exponent=1, 
                            copy = True, return_lapsym = False):
        if (not hasattr(self, 'laplacian_matrix') or self.laplacian_type != normed):
            self.laplacian_type = normed
            if self.affinity_matrix is None:
                self.affinity_matrix = self.get_affinity_matrix()
            if not return_lapsym or self.laplacian_type in ['symmetricnormalized', 'unnormalize']:
                self.laplacian_matrix = graph_laplacian(self.affinity_matrix, 
                                                        self.laplacian_type, 
                                                        symmetrize, scaling_epps, 
                                                        renormalization_exponent)
            else:
                (self.laplacian_matrix, 
                self.laplacian_symmetric, 
                self.w) = graph_laplacian(self.affinity_matrix, self.laplacian_type, 
                                        symmetrize, scaling_epps, renormalization_exponent,
                                        return_lapsym = True)
        if copy:
            return self.laplacian_matrix.copy()
        else:
            return self.laplacian_matrix
        
    # functions to assign distance, affinity, and Laplacian matrices:
        # the only checking done here is that they are square matrices
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
    
    def assign_laplacian_matrix(self, laplacian_matrix):
        (a, b) = laplacian_matrix.shape
        if a != b:
            raise ValueError("Laplacian matrix is not square")
        else:
            self.laplacian_matrix = laplacian_matrix
    def assign_neighbors_radius(self, radius):
        self.neighbors_radius = radius
