#pragma once

#include <opt/CombinedSolver.h>
#include "warp_field.hpp"

namespace kfusion {
    class WarpFieldOptimiser {
    public:
        WarpFieldOptimiser(WarpField *warp, CombinedSolver *solver);

        WarpFieldOptimiser(WarpField *warp, CombinedSolverParameters params);

        ~WarpFieldOptimiser() = default;

        void optimiseWarpData(const std::vector<Vec3f> &canonical_vertices,
                              const std::vector<Vec3f> &canonical_normals,
                              const std::vector<Vec3f> &live_vertices,
                              const std::vector<Vec3f> &live_normals);

    private:
        WarpField *warp_;
        CombinedSolver *solver_;
    };
}
