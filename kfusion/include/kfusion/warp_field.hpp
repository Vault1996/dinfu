#pragma once

#include <dual_quaternion.hpp>
#include <kfusion/types.hpp>
#include <nanoflann/nanoflann.hpp>
#include <knn_point_cloud.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>

#define KNN_NEIGHBOURS 8
namespace kfusion {
    typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, utils::PointCloud>,
            utils::PointCloud,
            3
    > kd_tree_t;


    /*!
     * \struct node
     * \brief Deformation node of a warp-field
     * \details The state of the warp field Wt at time t is defined by the values of a set of n
     * deformation nodes Nt_warp = {dg_v, dg_w, dg_se3}_t.
     *
     * \var node::vertex: Position of the vertex in space.
     * Equivalent to dg_v in the paper.
     *
     * \var node::transform: Transformation for each vertex to warp it into the live frame
     * Equivalent to dg_se3.
     *
     * \var node::radial basis weight
     * Equivalent to dg_w
     */
    struct deformation_node {
        Vec3f vertex;
        kfusion::utils::DualQuaternion<float> transform;
        float weight = 0;
    };

    class WarpField {
    public:
        WarpField();

        ~WarpField();

        void init(const cv::Mat &first_frame);

        void applyDQB(std::vector<Vec3f> &points, std::vector<Vec3f> &normals) const;

        utils::DualQuaternion<float> DQB(const Vec3f &vertex) const;

        void getWeightsAndUpdateKNN(const Vec3f &vertex, float weights[KNN_NEIGHBOURS]) const;

        float weighting(float squared_dist, float weight) const;

        void KNN(Vec3f point) const;

        void clear();

        const std::vector<deformation_node> *getNodes() const;

        std::vector<deformation_node> *getNodes();

        const cv::Mat getNodesAsMat() const;

        std::vector<float> *getDistSquared() const;

        std::vector<size_t> *getRetIndex() const;

        void buildKDTree();

    private:
        std::vector<deformation_node> *deformation_nodes_;
        kd_tree_t *kd_tree_index_;
        Affine3f warp_to_live_;
    };
}
