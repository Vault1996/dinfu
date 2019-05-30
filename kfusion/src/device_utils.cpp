#include <cstdio>
#include "kfusion/device_utils.hpp"
#include "safe_call.hpp"
#include <cuda.h>

int kfusion::cuda::getCudaEnabledDeviceCount() {
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);

    if (error == cudaErrorInsufficientDriver)
        return -1;

    if (error == cudaErrorNoDevice)
        return 0;

    cudaSafeCall(error);
    return count;
}

void kfusion::cuda::setDevice(int device) { cudaSafeCall(cudaSetDevice(device)); }

namespace {
    inline int convertSMVer2Cores(int major, int minor) {
        // Defines for GPU Architecture types (using the SM version to determine the #
        // of cores per SM
        typedef struct {
            int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM
            // minor version
            int Cores;
        } SMtoCores;

        SMtoCores gpuArchCoresPerSM[] = {{0x10, 8},
                                         {0x11, 8},
                                         {0x12, 8},
                                         {0x13, 8},
                                         {0x20, 32},
                                         {0x21, 48},
                                         {0x30, 192},
                                         {0x35, 192},
                                         {0x50, 128},
                                         {0x52, 128},
                                         {-1,   -1}};

        int index = 0;
        while (gpuArchCoresPerSM[index].SM != -1) {
            if (gpuArchCoresPerSM[index].SM == ((major << 4) + minor))
                return gpuArchCoresPerSM[index].Cores;
            index++;
        }
        printf("\nCan't determine number of cores. Unknown SM version %d.%d!\n", major, minor);
        return 0;
    }
}

bool kfusion::cuda::checkIfPreFermiGPU(int device) {
    if (device < 0)
        cudaSafeCall(cudaGetDevice(&device));

    cudaDeviceProp prop;
    cudaSafeCall(cudaGetDeviceProperties(&prop, device));
    return prop.major < 2;  // CC == 1.x
}

void kfusion::cuda::printShortCudaDeviceInfo(int device) {
    int count = getCudaEnabledDeviceCount();
    bool valid = (device >= 0) && (device < count);

    int beg = valid ? device : 0;
    int end = valid ? device + 1 : count;

    int driverVersion = 0, runtimeVersion = 0;
    cudaSafeCall(cudaDriverGetVersion(&driverVersion));
    cudaSafeCall(cudaRuntimeGetVersion(&runtimeVersion));

    for (int dev = beg; dev < end; ++dev) {
        cudaDeviceProp prop;
        cudaSafeCall(cudaGetDeviceProperties(&prop, dev));

        const char *arch_str = prop.major < 2 ? " (pre-Fermi)" : "";
        printf("Device %d:  \"%s\"  %.0fMb", dev, prop.name, (float) prop.totalGlobalMem / 1048576.0f);
        printf(", sm_%d%d%s, %d cores", prop.major, prop.minor, arch_str,
               convertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount);
        printf(", Driver/Runtime ver.%d.%d/%d.%d\n", driverVersion / 1000, driverVersion % 100, runtimeVersion / 1000,
               runtimeVersion % 100);
    }
    fflush(stdout);
}

