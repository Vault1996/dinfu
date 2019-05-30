#pragma once

#include <kfusion/types.hpp>
#include <vector>
#include <string>
#include <dual_quaternion.hpp>
#include <quaternion.hpp>
#include <kfusion/cuda/projective_icp.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/warp_field.hpp>
#include <kfusion/device_utils.hpp>
#include "warp_field_optimiser.hpp"

namespace kfusion {
    struct KF_EXPORTS KinFuParams {
        static KinFuParams default_params();

        static KinFuParams default_params_dynamicfusion();

        /**
         * In pixels
         */
        int cols;

        /**
         * In pixels
         */
        int rows;

        /**
         * Intrinsic parameters of a camera parameters
         */
        Intr intr;

        /**
         * Number of voxels
         */
        Vec3i volume_dims;

        /**
         * In meters
         */
        Vec3f volume_size;

        /**
         * Initial pose in meters
         */
        Affine3f volume_pose;

        /**
         * In meters
         */
        float bilateral_sigma_depth;

        /**
         * In pixels
         */
        float bilateral_sigma_spatial;

        /**
         * In pixels
         */
        int bilateral_kernel_size;

        /**
         * In meters
         */
        float icp_truncate_depth_dist;

        /**
         * In meters
         */
        float icp_dist_thres;
        /**
         * In radians
         */
        float icp_angle_thres;
        /**
         * In meters
         */
        std::vector<int> icp_iter_num;

        float tsdf_min_camera_movement;
        /**
         * In meters
         */
        float tsdf_trunc_dist;
        /**
         * Frames
         */
        int tsdf_max_weight;

        /**
         * In voxel sizes
         */
        float raycast_step_factor;

        /**
         * In voxel sizes
         */
        float gradient_delta_factor;

        /**
         * In meters
         */
        Vec3f light_pose;

    };

    class KF_EXPORTS KinFu {
    public:
        typedef cv::Ptr<KinFu> Ptr;

        KinFu(const KinFuParams &params);

        const KinFuParams &params() const;

        KinFuParams &params();

        const cuda::TsdfVolume &tsdf() const;

        cuda::TsdfVolume &tsdf();

        const cuda::ProjectiveICP &icp() const;

        cuda::ProjectiveICP &icp();

        const WarpField &getWarp() const;

        WarpField &getWarp();

        void reset();

        bool operator()(const cuda::Depth &depth, const cuda::Image &image = cuda::Image());

        void renderImage(cuda::Image &image, int flags = 0);

        void dynamicfusion(cuda::Depth &depth, cuda::Cloud &live_frame, cuda::Normals current_normals);

        void renderImage(cuda::Image &image, const Affine3f &pose, int flags = 0);

        Affine3f getCameraPose(int time = -1) const;

    private:
        void allocate_buffers();

        int frame_counter_;
        KinFuParams params_;

        std::vector<Affine3f> poses_;

        cuda::Dists dists_;
        cuda::Frame curr_, prev_, first_;

        cuda::Cloud points_;
        cuda::Normals normals_;
        cuda::Depth depths_;

        cv::Ptr<cuda::TsdfVolume> tsdf_volume_;
        cv::Ptr<cuda::ProjectiveICP> icp_;
        cv::Ptr<WarpField> warp_field_;
        cv::Ptr<WarpFieldOptimiser> optimiser_;
    };
}
