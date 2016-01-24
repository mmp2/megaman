/* Authors: Zhongyue Zhang <zhangz6@cs.washington.edu>

License: BSD 3 clause
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
        float target_precision) {
    int num_pts = dataset.size() / num_dims;
    dataset_ = new float[dataset.size()];
    std::copy(dataset.begin(), dataset.end(), dataset_);
    Matrix<float> data(dataset_, num_pts, num_dims);
    // TODO: add support for different distance metric.
    index_ = new Index< L2<float> >(data, AutotunedIndexParams(
            target_precision, 0.01, 0.1,0.1));
}

CyflannIndex::CyflannIndex(const std::vector<float>& dataset, int num_dims,
        float target_precision, std::string filename) {
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

void CyflannIndex::knnSearch(const std::vector<float>& queries,
        std::vector< std::vector<int> >& indices,
        std::vector< std::vector<float> >& dists,
        int knn, int num_dims) {
    int num_pts = queries.size() / num_dims;
    float array[queries.size()];
    std::copy(queries.begin(), queries.end(), array);
    Matrix<float> qpts(array, num_pts, num_dims);
    index_->knnSearch(qpts, indices, dists, knn, SearchParams());
}

int CyflannIndex::radiusSearch(const std::vector<float>& queries,
        std::vector< std::vector<int> >& indices,
        std::vector< std::vector<float> >& dists,
        float radius, int num_dims) {
    int num_pts = queries.size() / num_dims;
    float array[queries.size()];
    std::copy(queries.begin(), queries.end(), array);
    Matrix<float> dataset(array, num_pts, num_dims);
    int res = index_->radiusSearch(dataset, indices, dists, radius, SearchParams());
    return res;
}

void CyflannIndex::save(std::string filename) {
    index_->save(filename);
}

int CyflannIndex::veclen() { return index_->veclen(); }

int CyflannIndex::size() { return index_->size(); }
