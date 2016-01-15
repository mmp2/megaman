/* Authors: Zhongyue Zhang <zhangz6@cs.washington.edu>

License: BSD 3 clause
================================================= */
#ifndef FLANN_RADIUS_NEIGHBORS_H_
#define FLANN_RADIUS_NEIGHBORS_H_

#include <flann/flann.hpp> 
#include <vector>
using namespace flann;

//using namespace flann;

// Takes a flattened matrix queries, with dimension num_dims.
// For each data point in queries, search for neighbors within the radius.
int radiusSearch(const std::vector<float>& queries,
		 std::vector< std::vector<int> >& indices,
                 std::vector< std::vector<float> >& dists,
		 float radius, int num_dims);

#endif // FLANN_RADIUS_NEIGHBORS_H_
