# distutils: language=c++

# Authors: Zhongyue Zhang <zhangz6@cs.washington.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

from __future__ import division
from scipy import sparse
import numpy as np
cimport numpy as np
from index cimport *

cdef class Index:
    """
    Wrapper for flann c++ index class
    """
    cdef CyflannIndex* _thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, np.ndarray[double, ndim=2] dataset,
            target_precision=None, saved_index=None, index_type=None,
            num_trees=4, branching=32, iterations=11, cb_index=0.2,
            build_weight=0.01, memory_weight=0, sample_fraction=0.1):
        """
        Constucts a index class. The default index is kmeans index. If target
        precision is specified, the index will be autotuned. If saved index is
        specified, the index will be loaded from the saved file. index_type can
        be either "kdtrees", "kmeans" or "composite".
        kdtrees has parameters: num_trees
        kmeans has parameters: branching, iterations, cb_index
        composite combines kdtrees and kmeans
        """
        if target_precision is not None:
            self._thisptr = new CyflannIndex(dataset.flatten(),
                    dataset.shape[1], target_precision, build_weight,
                    memory_weight, sample_fraction)
        elif saved_index is not None:
            # setting the target precision is purely a way to get arround
            # function overloading. I cannot get it to work without it.
            self._thisptr = new CyflannIndex(dataset.flatten(),
                    dataset.shape[1], saved_index)
        elif index_type is not None:
            index_type = index_type.encode('utf-8')
            self._thisptr = new CyflannIndex(dataset.flatten(),
                    dataset.shape[1], index_type, num_trees, branching,
                    iterations, cb_index)
        else:
            self._thisptr = new CyflannIndex(dataset.flatten(),
                    dataset.shape[1])

    def __dealloc__(self):
        del self._thisptr

    def buildIndex(self):
        """
        Builds a index of the data for future queries.
        """
        self._thisptr.buildIndex()

    def knn_neighbors_graph(self, np.ndarray[double, ndim=2] X, int knn,
            int num_checks=48):
        if knn < 1:
            raise ValueError('neighbors_radius must be >=0.')
        cdef int nsam, ndim
        nsam = X.shape[0]
        ndim = X.shape[1]
        X = np.require(X, requirements = ['A', 'C']) # required for FLANN
        cdef vector[vector[dtypei_t]] indices;
        cdef vector[vector[dtype_t]] dists;
        cdef int res = self._thisptr.knnSearch(X.flatten(), indices, dists,
                knn, ndim, num_checks)
        lengths = [len(nghbr_idx) for nghbr_idx in indices]
        indpts = list(np.cumsum(lengths))
        indpts.insert(0,0)
        indpts = np.array(indpts)
        # aparrently cython does not support iterating over vector withh
        # size = 1
        cdef np.ndarray[dtype_t, ndim=1] data = np.zeros(res, dtype=np.float32)
        cdef np.ndarray[dtypei_t, ndim=1] indices_list = np.zeros(
                res, dtype=np.int32)
        self.fillDataAndIndices(data, indices_list, dists, indices)
        # self.size() is slower than self._thisptr.size(). def methods are
        # are usually quite a bit slower than cdef methods.
        graph = sparse.csr_matrix((data, indices_list, indpts), shape = (nsam,
                                  self._thisptr.size()))
        graph.data = np.sqrt(graph.data) # FLANN returns squared distance
        return graph

    def radius_neighbors_graph(self, np.ndarray[double, ndim=2] X,
            float radius, int num_checks=32):
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
        num_checks: specifying the number of times the tree(s) in the index
            should be recursively traversed. A higher value for this parameter
            would give better search precision, but also take more time.

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
        cdef int res = self._thisptr.radiusSearch(X.flatten(), indices, dists,
                radius, ndim, num_checks)
        lengths = [len(nghbr_idx) for nghbr_idx in indices]
        indpts = list(np.cumsum(lengths))
        indpts.insert(0,0)
        indpts = np.array(indpts)
        # aparrently cython does not support iterating over vector withh
        # size = 1
        cdef np.ndarray[dtype_t, ndim=1] data = np.zeros(res, dtype=np.float32)
        cdef np.ndarray[dtypei_t, ndim=1] indices_list = np.zeros(
                res, dtype=np.int32)
        self.fillDataAndIndices(data, indices_list, dists, indices)
        # self.size() is slower than self._thisptr.size(). def methods are
        # are usually quite a bit slower than cdef methods.
        graph = sparse.csr_matrix((data, indices_list, indpts), shape = (nsam,
                                  self._thisptr.size()))
        graph.data = np.sqrt(graph.data) # FLANN returns squared distance
        return graph

    def save(self, string filename):
        """
        Saves the index to a file.
        """
        self._thisptr.save(filename)

    def veclen(self):
        """
        Length of a single data point
        """
        return self._thisptr.veclen()

    def size(self):
        """
        Returns the number of data points in the current dataset.
        """
        return self._thisptr.size()

    cdef void fillDataAndIndices(self, np.ndarray[dtype_t, ndim=1] data,
                            np.ndarray[dtypei_t, ndim=1] indices_list,
                            vector[vector[dtype_t]] dists,
                            vector[vector[dtypei_t]] indices):
        cdef int i, j, k
        k = 0
        for i in xrange(indices.size()):
            for j in xrange(indices[i].size()):
                data[k] = dists[i][j]
                indices_list[k] = indices[i][j]
                k += 1
