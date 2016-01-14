# Authors: Zhongyue Zhang <zhangz6@cs.washington.edu>
# License: BSD 3 clause                   

from __future__ import division
import cython
from libcpp.vector cimport vector

cdef extern from "flann_radius_neighbors.h":
    int radiusSearch(const vector[float]& queries,
                 vector[vector[int]]& indices,
                 vector[vector[float]]& dists,
                 float radius, int num_dims)
                         