#pragma once

#include <kfusion/exports.hpp>

namespace kfusion {
    namespace cuda {
        KF_EXPORTS void printShortCudaDeviceInfo(int device);

        KF_EXPORTS bool checkIfPreFermiGPU(int device);

        KF_EXPORTS int getCudaEnabledDeviceCount();

        KF_EXPORTS void setDevice(int device);
    }
}
