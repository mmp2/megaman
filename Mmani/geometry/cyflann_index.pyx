# Author: Zhongyue Zhang <zhangz6@cs.washington.edu>
# License: BSD 3 clause

import numpy as np
from scipy import sparse
cimport numpy as np
from cyflann_index cimport radiusSearch

ctypedef np.float32_t dtype_t
ctypedef np.int32_t dtypei_t

cdef class cyflann_index:
    """
    partial wrapper for flann index class.
    """    
    
    def radius_neighbors_graph(self, X, radius):
        """
        Constructs a sparse distance matrix called graph in coo
        format. 
        Parameters
        ----------
        X: data matrix, array_like, shape = (n_samples, n_dimensions )
        radius: neighborhood radius, scalar
            the neighbors lying approximately within radius of a node will
            be returned. Or, in other words, all distances will be less or 
            equal to radius. There will be entries in the matrix for zero 
            distances.
            
            Attention when converting to dense: The rest of the distances
            should not be considered 0, but "large".
        
        Returns
        -------
        graph: the distance matrix, array_like, shape = (X.shape[0],X.shape[0])
               sparse csr format
        
        Notes
        -----
        With approximate neiborhood search, the matrix is not necessarily 
        symmetric. 
        """
        
        if radius < 0.:
            raise ValueError('neighbors_radius must be >=0.')
        radius *= radius
        cdef vector[vector[int]] indices;
        cdef vector[vector[cython.float]] dists;
        cdef int nsam, ndim
        nsam, ndim = X.shape
        cdef int res = radiusSearch(X.flatten(), indices, dists, radius, ndim)
        X = np.require(X, requirements = ['A', 'C']) # required for FLANN
        lengths = [len(nghbr_idx) for nghbr_idx in indices]
        indpts = list(np.cumsum(lengths))
        indpts.insert(0,0)
        indpts = np.array(indpts)
        
        # aparrently cython does not support iterating over vector withh 
        # size = 1
        cdef int i, j, k
        k = 0
        cdef np.ndarray[dtype_t, ndim=1] data = np.zeros(res, dtype=np.float32)
        cdef np.ndarray[dtypei_t, ndim=1] indices_list = np.zeros(
                res, dtype=np.int32)
        data = np.zeros(res, dtype=np.float32)
        indices_list = np.zeros(res, dtype=np.int32)
        for i in xrange(indices.size()):
            for j in xrange(indices[i].size()):
                data[k] = dists[i][j]
                indices_list[k] = indices[i][j]
                k += 1
            #for j in xrange(indices[i].size()):
        graph = sparse.csr_matrix((data, indices_list, indpts), shape = (nsam, nsam))
        graph.data = np.sqrt(graph.data) # FLANN returns squared distance
        return graph