# Authors: Zhongyue Zhang <zhangz6@cs.washington.edu>
# License: BSD 3 clause                   

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
        CyflannIndex(const vector[dtype_t]& dataset, dtypei_t ndim, 
                     dtype_t target_precision)
        CyflannIndex(const vector[dtype_t]& dataset, dtypei_t ndim,
                     dtype_t target_precision, string filename)
        void buildIndex()
        void knnSearch(const vector[dtype_t]& queries,
            vector[vector[dtypei_t]]& indices,
            vector[vector[dtype_t]]& dists,
            dtypei_t knn, dtypei_t num_dims)
        int radiusSearch(const vector[dtype_t]& queries,
            vector[vector[dtypei_t]]& indices,
            vector[vector[dtype_t]]& dists,
            dtype_t radius, dtypei_t num_dims)
        void save(string filename)
        int veclen()
        int size()
        