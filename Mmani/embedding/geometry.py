"""
Geometric (Riemannian) Manifold learning utilities and algorithms

Graphs are represented with their weighted adjacency matrices, preferably using
sparse matrices.
"""
#Authors: Marina Meila <mmp@stat.washington.edu>
#          Jake Vanderplas <vanderplas@astro.washington.edu>
# License: BSD 3 clause

import numpy as np
from scipy import sparse

from .graph_shortest_path import graph_shortest_path

###############################################################################
# Path and connected component analysis.
# Code adapted from networkx
# Code from sklean/graph
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
    >>> from sklearn.utils.graph import single_source_shortest_path_length
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


if hasattr(sparse, 'connected_components'):
    connected_components = sparse.connected_components
else:
    from .sparsetools import connected_components


###############################################################################
# Graph laplacian
# Code adapted from the Matlab package XXX of Dominique Perrault-Joncas
def graph_laplacian(csgraph, normed='geometric', symmetrize=True, scaling_epps=0., renormalization_exponent=1, return_diag=False):
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
    3) the diagonal of lap is no longer set to 0
    4) if csgraph is not symmetric the out-degree is used in the
    computation and no warning is raised. 
    However, it is not recommended to use this function for directed graphs.
    Use directed_laplacian() (NYImplemented) instead
    """
    if csgraph.ndim != 2 or csgraph.shape[0] != csgraph.shape[1]:
        raise ValueError('csgraph must be a square matrix or array')

    ## what is this anyways?? 
    normed = normed.lower()
    if normed not in ('unnormalized', 'geometric', 'randomwalk', 'symmetricnormalized','renormalized' ):
        raise ValueError('normed must be one of unnormalized, geometric, randomwalk, symmetricnormalized, renormalized')
    if (np.issubdtype(csgraph.dtype, np.int) or np.issubdtype(csgraph.dtype, np.uint)):
        csgraph = csgraph.astype(np.float)

    if sparse.isspmatrix(csgraph):
        return _laplacian_sparse(csgraph, normed=normed, symmetrize=symmetrize, scaling_epps=scaling_epps, renormalization_exponent=renormalization_exponent, return_diag=return_diag)

    else:
        return _laplacian_dense(csgraph, normed=normed, symmetrize=symmetrize, scaling_epps=scaling_epps, renormalization_exponent=renormalization_exponent, return_diag=return_diag)


def _laplacian_sparse(csgraph, normed='geometric', symmetrize=True, scaling_epps=0., renormalization_exponent=1, return_diag=False):
    ## what is thi s ?
    n_nodes = graph.shape[0]
    if not graph.format == 'coo':
        lap = graph.tocoo()
    else:
        lap = graph.copy()
    if symmetrize:
        lap += lap.T   
        lap.data /= 2.
    diag_mask = (lap.row == lap.col)
    if not diag_mask.sum() == n_nodes:  ##whayt?? whoy not .size ?
        # The sparsity pattern of the matrix has holes on the diagonal,
        # we need to fix that
        diag_idx = lap.row[diag_mask]
        diagonal_holes = list(set(range(n_nodes)).difference(diag_idx))
        new_data = np.concatenate([lap.data, np.ones(len(diagonal_holes))])
        new_row = np.concatenate([lap.row, diagonal_holes])
        new_col = np.concatenate([lap.col, diagonal_holes])
        lap = sparse.coo_matrix((new_data, (new_row, new_col)),
                                shape=lap.shape)
        diag_mask = (lap.row == lap.col)

    #lap.data[diag_mask] = 0  #why is this yere
    degrees = np.asarray(lap.sum(axis=1)).squeeze()
    if normed == 'symmetricnormalized':
        w = np.sqrt(degrees)
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap.data /= w[lap.row]
        lap.data /= w[lap.col]
        lap.data[diag_mask] -= 1. 
# whya ll this(w_zeros[lap.row[diag_mask]]).astype(lap.data.dtype-1.)
    if normed == 'geometric':
        w = degrees.copy()     # normzlize one symmetrically by d
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap.data /= w[lap.row]
        lap.data /= w[lap.col]
        w = np.asarray(lap.sum(axis=1)).squeeze() #normalize again asymmetricall
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
        lap.data /= w[lap.row]
        lap.data[diag_mask] -= 1.

    if normed == 'unnormalized':
        lap[diagmask] -= degrees[lap.row]
    if normed == 'randomwalk':
        lap /= degrees[lap.row]

    if scaling_epps > 0.:
        lap *= 4/np.sqrt(scaling_epps)

    if return_diag:
        return lap, np.asarray( lap[diagmask] ).squeeze()
    return lap

# TO BE UPDATED
def _laplacian_dense(csgraph, normed='geometric', symmetrize=True, scaling_epps=0., renormalization_exponent=1, return_diag=False):
    n_nodes = graph.shape[0]
    lap = -np.asarray(graph)  # minus sign leads to a copy
    # set diagonal to zero
    lap.flat[::n_nodes + 1] = 0
    w = -lap.sum(axis=0)
    if normed:
        w = np.sqrt(w)
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap /= w
        lap /= w[:, np.newaxis]
        lap.flat[::n_nodes + 1] = (1 - w_zeros).astype(lap.dtype)
    else:
        lap.flat[::n_nodes + 1] = w.astype(lap.dtype)

    if return_diag:
        return lap, w
    return lap

def distance_matrix( X, adjacency='radius_neighbors', neighbor_radius=None,
                     n_neighbors=0 ):
    # DNearest neighbors has issues. TB FIXED
    if mode == 'nearest_neighbors':
        warnings.warn("Nearest neighbors currently does not work"
                      "falling back to radius neighbors")
        mode = 'radius_neighbors'

    if mode == 'radius_neighbors':
        neighbors_radius_ = (neighbors_radius
                             if neighbors_radius is not None else 1.0 / X.shape[1])   # to put another defaault value, like diam(X)/sqrt(dimensions)/10
        distance_matrix = radius_neighbors_graph(X, neighbors_radius_, mode='distance')
        return distance_matrix


class DistanceMatrix:

    def __init__(self, mode="radius_neighbors",
                 gamma=None, neighbors_radius = None, n_neighbors=None):
        self.mode = mode
        self.gamma = gamma
        self.neighbors_radius = neighbors_radius
        self.n_neighbors = n_neighbors

    @property
    def _pairwise(self):
        return self.mode == "precomputed"

    def _get_distance_matrix_(self, X):
        if self.mode == 'precomputed':
            self.distance_matrix = X
        else:
            self.distance_matrix = distance_matrix(X, mode=self.mode, neighbors_radius=self.neighbors_radius, n_neighbors=self.n_neighbors)
        return self.distance_matrix

    def get_distance_matrix( self, X, copy=True ):
        if self.distance_matrix is None:
            self.distance_matrix = distance_matrix(X, mode=self.mode, neighbors_radius=self.neighbors_radius, n_neighbors=self.n_neighbors)
        if copy:
            return self.distance_matrix_.copy()
        else:
            return self.distance_matrix_

def affinity_matrix( distances, neighbor_radius ):
    if neighbor_radius not >0.:
        raise ValueError('neighbor_radius must be >0.')
    A = distances.copy()
    if sparse.isspmatrix( A ):
        A.data **= 2
        A.data /= -neighbors_radius_**2
        np.exp( A.data, A_.data )
    else:
        A **= 2
        A /= -neighbors_radius_**2
        np.exp(A, A)
    return A
