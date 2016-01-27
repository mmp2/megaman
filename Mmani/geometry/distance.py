# Authors: Marina Meila <mmp@stat.washington.edu>
#         James McQueen <jmcq@u.washington.edu>
# License: BSD 3 clause
from __future__ import division ## removes integer division
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist
from Mmani.geometry.cyflann.index import Index
import subprocess, os, sys, warnings

def _row_col_from_condensed_index(N,compr_ind):
    # convert from pdist compressed index format to (I, J) (upper triangular) pairs.
    b = 1 -2*N 
    i = np.floor((-b - np.sqrt(b**2 - 8*compr_ind))/2)
    j = compr_ind + i*(b + i + 2)/2 + 1
    return (i,j)  

def distance_matrix(X, method = 'auto', flindex = None, radius = None, cyindex = None):
    """
    Computes pairwise distance matrix. Interface function.
    
    Parameters
    ----------
    X : data matrix, array_like, shape = (n_samples, n_dimensions)
    method : one of 'auto', 'brute', 'pyflann', 'cyflann', or 'cython'.
        'pyflann' requires python library pyflann
        'cython' requires FLANN and cython 
        'cyflann' requires a UNIX system 
    flindex : a pyflann pre-computed flindex 
    radius : neighborhood radius, scalar
        the neighbors lying approximately within radius of a node will
        be returned. Or, in other words, all distances will be less or equal
        to radius. There will be entries in the matrix for zero distances.
        Attention when converting to dense: The rest of the distances
        should not be considered 0, but "large".
    cyindex : A cython computed FLANN index. 
    
    Returns
    -------
    graph : the distance matrix, array_like, shape (X.shape[0]. X.shape[0])
           sparse csr_format. Zeros on the diagonal are true zeros. 
           Zeros not on the diagonal should be considered infinite     
    """
    if radius is None:
        radius = 1/X.shape[1]
    if method == 'auto':
        # change this based on benchmarking results
        method = 'brute'
    if method == 'pyflann':
        if flindex is None:
            raise ValueError('must pass a flindex when using pyflann')
            return None
        else:
            graph = fl_radius_neighbors_graph(X, radius, flindex)
    elif method == 'cyflann':
        graph = fl_cpp_radius_neighbors_graph(X, radius)
    elif method == 'cython':
        if cyindex is None:
            cyindex = Index(X)
        cyindex.buildIndex()
        graph = cyindex.radius_neighbors_graph(X, radius)
    elif method == 'brute':
        graph = radius_neighbors_graph(X, radius)
    else: 
        raise ValueError('method must be one of: (auto, brute, pyflann, cyflann)')
        return None
    return graph
        
def radius_neighbors_graph(X, radius):
    """
    Computes pairwise distance matrix using dense method. 
    
    Parameters
    ----------
    X: data matrix, array_like, shape = (n_samples, n_dimensions)
    radius: neighborhood radius, scalar
    
    Returns
    -------
    graph: the distance matrix, array_like, shape (X.shape[0]. X.shape[0])
           sparse csr_format. Zeros on the diagonal are true zeros. 
           Zeros not on the diagonal should be considered infinite 
    """
    N = X.shape[0]
    all_dists = pdist(X)
    compr_ind = np.where(all_dists <= radius)[0]
    (I, J) = _row_col_from_condensed_index(X.shape[0],compr_ind)
    graph = sparse.coo_matrix((all_dists[compr_ind], (I, J)), shape = (N, N))
    graph = graph + graph.T #  symmetrize distance matrix (converts to csr)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # sparse will complain this is faster with lil_matrix:
        graph.setdiag(0.0) # diagonal values are true zeros, otherwise they're infinite
    return(graph)
        
def fl_cpp_radius_neighbors_graph(X, radius):
    """
    Constructs a sparse distance matrix called graph in coo
    format using pre-compiled C++ function.
    
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
    radius *= radius # FLANN computes squared distance. 
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
    graph.data = np.sqrt(graph.data) # FLANN returns squared distance
    return graph
    
def fl_radius_neighbors_graph(X, radius, flindex):
    """
    Constructs a sparse distance matrix called graph in coo format using pyflann.
    
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
    radius *= radius
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
    graph.data = np.sqrt(graph.data) # FLANN returns squared distance
    return graph
