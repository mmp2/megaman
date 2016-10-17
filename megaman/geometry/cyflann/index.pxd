# Authors: Zhongyue Zhang <zhangz6@cs.washington.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

from __future__ import division
import cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string

ctypedef np.float32_t dtype_t
ctypedef np.int32_t dtypei_t

cdef extern from "cyflann_index.h":
    cdef cppclass CyflannIndex:
        CyflannIndex(const vector[dtype_t]& dataset, dtypei_t ndim) except +
        CyflannIndex(const vector[dtype_t]& dataset, dtypei_t num_dims,
                string index_type, dtypei_t num_trees, dtypei_t branching,
                dtypei_t iterations, dtype_t cb_index)
        CyflannIndex(const vector[dtype_t]& dataset, dtypei_t ndim,
                dtype_t target_precision, dtype_t build_weight,
                dtype_t memory_weight, dtype_t sample_fraction)
        CyflannIndex(const vector[dtype_t]& dataset, dtypei_t ndim,
                string filename)
        void buildIndex()
        int knnSearch(const vector[dtype_t]& queries,
            vector[vector[dtypei_t]]& indices,
            vector[vector[dtype_t]]& dists,
            dtypei_t knn, dtypei_t num_dims, dtypei_t num_checks)
        int radiusSearch(const vector[dtype_t]& queries,
            vector[vector[dtypei_t]]& indices,
            vector[vector[dtype_t]]& dists,
            dtype_t radius, dtypei_t num_dims, dtypei_t num_checks)
        void save(string filename)
        int veclen()
        int size()
