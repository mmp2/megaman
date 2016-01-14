#include <flann/flann.hpp> 
#include <vector>
using namespace flann;

int radiusSearch(const std::vector<float>& queries,
		 std::vector< std::vector<int> >& indices,
                 std::vector< std::vector<float> >& dists,
                 float radius, int num_dims) {
  int num_pts = queries.size() / num_dims;
  Matrix<float> dataset(new float[num_pts*num_dims], num_pts, num_dims);
  for(int n = 0; n < num_pts; n++){
      for(int d = 0; d < num_dims; d++){
	dataset[n][d]=queries[num_dims * n + d];
      }
  }
  // TODO(zhongyue): add support for different distance metric.
  Index<L2<float> > index(dataset, KMeansIndexParams());
  index.buildIndex();
  return index.radiusSearch(dataset, indices, dists, radius, SearchParams());
}
