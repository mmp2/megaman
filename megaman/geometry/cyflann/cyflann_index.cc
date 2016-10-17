/* Authors: Zhongyue Zhang <zhangz6@cs.washington.edu>

LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE
================================================= */

#include "cyflann_index.h"

CyflannIndex::CyflannIndex(const std::vector<float>& dataset, int num_dims) {
    int num_pts = dataset.size() / num_dims;
    dataset_ = new float[dataset.size()];
    std::copy(dataset.begin(), dataset.end(), dataset_);
    Matrix<float> data(dataset_, num_pts, num_dims);
    // TODO: add support for different distance metric.
    index_ = new Index< L2<float> >(data, KMeansIndexParams());
}

CyflannIndex::CyflannIndex(const std::vector<float>& dataset, int num_dims,
        std::string index_type, int num_trees, int branching, int iterations,
        float cb_index) {
    int num_pts = dataset.size() / num_dims;
    dataset_ = new float[dataset.size()];
    std::copy(dataset.begin(), dataset.end(), dataset_);
    Matrix<float> data(dataset_, num_pts, num_dims);
    // TODO: wrap all info into a class in the future.
    if (index_type == "kdtrees") {
        index_ = new Index< L2<float> >(data, KDTreeIndexParams(num_trees));
    } else if (index_type == "kmeans") {
        index_ = new Index< L2<float> >(data, KMeansIndexParams(branching,
                iterations,  FLANN_CENTERS_RANDOM, cb_index));
    } else {
        index_ = new Index< L2<float> >(data, CompositeIndexParams(num_trees,
                branching, iterations,  FLANN_CENTERS_RANDOM, cb_index));
    }
}

CyflannIndex::CyflannIndex(const std::vector<float>& dataset, int num_dims,
        float target_precision, float build_weight, float memory_weight,
        float sample_fraction) {
    int num_pts = dataset.size() / num_dims;
    dataset_ = new float[dataset.size()];
    std::copy(dataset.begin(), dataset.end(), dataset_);
    Matrix<float> data(dataset_, num_pts, num_dims);
    // TODO: add support for different distance metric.
    index_ = new Index< L2<float> >(data, AutotunedIndexParams(
            target_precision, build_weight, memory_weight, sample_fraction));
}

CyflannIndex::CyflannIndex(const std::vector<float>& dataset, int num_dims,
        std::string filename) {
    int num_pts = dataset.size() / num_dims;
    dataset_ = new float[dataset.size()];
    std::copy(dataset.begin(), dataset.end(), dataset_);
    Matrix<float> data(dataset_, num_pts, num_dims);
    // TODO: add support for different distance metric.
    index_ = new Index< L2<float> >(data, SavedIndexParams(filename));
}

CyflannIndex::~CyflannIndex() {
    delete index_;
    delete[] dataset_;
}

void CyflannIndex::buildIndex(){
    index_->buildIndex();
}

int CyflannIndex::knnSearch(const std::vector<float>& queries,
        std::vector< std::vector<int> >& indices,
        std::vector< std::vector<float> >& dists,
        int knn, int num_dims, int num_checks) {
    int num_pts = queries.size() / num_dims;
    float* array = new float[queries.size()];
    std::copy(queries.begin(), queries.end(), array);
    Matrix<float> qpts(array, num_pts, num_dims);
    int res = index_->knnSearch(qpts, indices, dists, knn,
        SearchParams(num_checks));
    delete[] array;
    return res;
}

int CyflannIndex::radiusSearch(const std::vector<float>& queries,
        std::vector< std::vector<int> >& indices,
        std::vector< std::vector<float> >& dists,
        float radius, int num_dims, int num_checks) {
    int num_pts = queries.size() / num_dims;
    float* array = new float[queries.size()];
    std::copy(queries.begin(), queries.end(), array);
    Matrix<float> dataset(array, num_pts, num_dims);
    int res = index_->radiusSearch(dataset, indices, dists, radius,
        SearchParams(num_checks));
    delete[] array;
    return res;
}

void CyflannIndex::save(std::string filename) {
    index_->save(filename);
}

int CyflannIndex::veclen() { return index_->veclen(); }

int CyflannIndex::size() { return index_->size(); }
