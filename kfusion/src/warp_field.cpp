#include <dual_quaternion.hpp>
#include <knn_point_cloud.hpp>
#include <nanoflann.hpp>
#include "kfusion/warp_field.hpp"
#include "internal.hpp"
#include "precomp.hpp"
#include <kfusion/optimisation.hpp>
#include <opt/main.h>

using namespace kfusion;

utils::PointCloud cloud;
nanoflann::KNNResultSet<float> *resultSet_;
std::vector<float> out_dist_sqr_;
std::vector<size_t> ret_index_;

WarpField::WarpField() {
    deformation_nodes_ = new std::vector<deformation_node>();
    kd_tree_index_ = new kd_tree_t(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    ret_index_ = std::vector<size_t>(KNN_NEIGHBOURS);
    out_dist_sqr_ = std::vector<float>(KNN_NEIGHBOURS);
    resultSet_ = new nanoflann::KNNResultSet<float>(KNN_NEIGHBOURS);
    resultSet_->init(&ret_index_[0], &out_dist_sqr_[0]);
    warp_to_live_ = cv::Affine3f();
}

/**
 * Initialize deformation nodes with KD-tree with
 */
void WarpField::init(const cv::Mat &first_frame) {
    deformation_nodes_->resize(first_frame.cols * first_frame.rows);

    // Initialize sparsely
    int step = 50;
    for (size_t i = 0; i < first_frame.rows; i += step) {
        for (size_t j = 0; j < first_frame.cols; j += step) {
            auto point = first_frame.at<Point>(i, j);
            if (std::isnan(point.x)) {
                continue;
            }

            deformation_nodes_->at(i * first_frame.cols + j).transform = utils::DualQuaternion<float>();
            deformation_nodes_->at(i * first_frame.cols + j).vertex = Vec3f(point.x, point.y, point.z);
            deformation_nodes_->at(i * first_frame.cols + j).weight = 3.f;
        }
    }

    buildKDTree();
}

/**
 * Apply Dual Quaternion Blending both on points and normals
 */
void WarpField::applyDQB(std::vector<Vec3f> &points, std::vector<Vec3f> &normals) const {
    int i = 0;
    for (auto &point : points) {
        if (std::isnan(point[0]) || std::isnan(normals[i][0]))
            continue;
        utils::DualQuaternion<float> dqb = DQB(point);
        dqb.transform(point);
        point = warp_to_live_ * point;

        dqb.transform(normals[i]);
        normals[i] = warp_to_live_ * normals[i];
        i++;
    }
}

utils::DualQuaternion<float> WarpField::DQB(const Vec3f &vertex) const {
    float weights[KNN_NEIGHBOURS];
    getWeightsAndUpdateKNN(vertex, weights);
    utils::Quaternion<float> translation_sum(0, 0, 0, 0);
    utils::Quaternion<float> rotation_sum(0, 0, 0, 0);
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++) {
        translation_sum += weights[i] * deformation_nodes_->at(ret_index_[i]).transform.getTranslation();
        rotation_sum += weights[i] * deformation_nodes_->at(ret_index_[i]).transform.getRotation();
    }
    rotation_sum.normalize();
    auto res = utils::DualQuaternion<float>(translation_sum, rotation_sum);
    return res;
}

void WarpField::getWeightsAndUpdateKNN(const Vec3f &vertex, float weights[KNN_NEIGHBOURS]) const {
    KNN(vertex);
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++) {
        weights[i] = weighting(out_dist_sqr_[i], deformation_nodes_->at(ret_index_[i]).weight);
    }
}

float WarpField::weighting(float squared_dist, float weight) const {
    return (float) exp(-squared_dist / (2 * weight * weight));
}

void WarpField::KNN(Vec3f point) const {
    resultSet_->init(&ret_index_[0], &out_dist_sqr_[0]);
    kd_tree_index_->findNeighbors(*resultSet_, point.val, nanoflann::SearchParams(10));
}

const std::vector<deformation_node> *WarpField::getNodes() const {
    return deformation_nodes_;
}

std::vector<deformation_node> *WarpField::getNodes() {
    return deformation_nodes_;
}

void WarpField::buildKDTree() {
    //    Build KD-tree with current warp nodes.
    cloud.pts.resize(deformation_nodes_->size());
    for (size_t i = 0; i < deformation_nodes_->size(); i++) {
        cloud.pts[i] = deformation_nodes_->at(i).vertex;
    }
    kd_tree_index_->buildIndex();
}

const cv::Mat WarpField::getNodesAsMat() const {
    cv::Mat matrix(1, deformation_nodes_->size(), CV_32FC3);
    for (int i = 0; i < deformation_nodes_->size(); i++) {
        auto &node = matrix.at<cv::Vec3f>(i);
        deformation_nodes_->at(i).transform.getTranslation(node);
        node += deformation_nodes_->at(i).vertex;
    }
    return matrix;
}

void WarpField::clear() {}

std::vector<float> *WarpField::getDistSquared() const {
    return &out_dist_sqr_;
}

std::vector<size_t> *WarpField::getRetIndex() const {
    return &ret_index_;
}

WarpField::~WarpField() {
    delete deformation_nodes_;
    delete resultSet_;
    delete kd_tree_index_;
}
