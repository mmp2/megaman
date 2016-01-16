# Authors: Zhongyue Zhang <zhangz6@cs.washington.edu>
# License: BSD 3 clause

from __future__ import division
from scipy import sparse
import numpy as np
cimport numpy as np
from index cimport *

cdef class Index:
    """
    wrapper for flann c++ index class
    """
    cdef CyflannIndex* _thisptr      # hold a C++ instance which we're wrapping
    
    def __cinit__(self, np.ndarray[double, ndim=2] dataset, 
                  target_precision=None):
        if target_precision is None:
            self._thisptr = new CyflannIndex(dataset.flatten(),
                    dataset.shape[1])
        else:
            self._thisptr = new CyflannIndex(dataset.flatten(), 
                    dataset.shape[1], target_precision)
    
    def __dealloc__(self):
        del self._thisptr
    
    def buildIndex(self):
        self._thisptr.buildIndex()
    
    def knn_neighbors_graph(self, np.ndarray[double, ndim=2] X, int knn):
        if knn < 1:
            raise ValueError('neighbors_radius must be >=0.')
        raise NotImplementedError("knnSearch is not yet implemented.")
        return None
    
    def radius_neighbors_graph(self, np.ndarray[double, ndim=2] X,
            float radius):
        """
        Constructs a sparse distance matrix called graph in csr
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
        graph: the distance matrix, array_like, shape = (# of data points in
               dataset, # of data points in queries) sparse csr format
        
        Notes
        -----
        With approximate neiborhood search, the matrix is not necessarily 
        symmetric. 
        """
        if radius < 0.:
            raise ValueError('neighbors_radius must be >=0.')
        radius *= radius
        cdef int nsam, ndim
        nsam = X.shape[0]
        ndim = X.shape[1]
        X = np.require(X, requirements = ['A', 'C']) # required for FLANN
        cdef vector[vector[dtypei_t]] indices;
        cdef vector[vector[dtype_t]] dists;
        cdef int res = self._thisptr.radiusSearch(X.flatten(), indices, dists, radius, ndim)
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
        for i in xrange(indices.size()):
            for j in xrange(indices[i].size()):
                data[k] = dists[i][j]
                indices_list[k] = indices[i][j]
                k += 1
        # self.size() is slower than self._thisptr.size(). def methods are
        # are usually quite a bit slower than cdef methods.
        graph = sparse.csr_matrix((data, indices_list, indpts), shape = (nsam, 
                                  self._thisptr.size()))
        graph.data = np.sqrt(graph.data) # FLANN returns squared distance
        return graph
    
    def save(self, string filename):
        self._thisptr.save(filename)
    
    def veclen(self):
        """
        length of a single datapoint
        """
        return self._thisptr.veclen()
    
    def size(self):
        return self._thisptr.size()
