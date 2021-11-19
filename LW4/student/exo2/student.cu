#include <iostream>
#include <exo2/student.h>
#include <OPP_cuda.cuh>

#define SIZE_HISTO 256.f

namespace {
    __global__
    void computeHistogram(unsigned *out, float *in, unsigned size) {
        unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= size) return;
        // compute histogram position
        unsigned pos = in[tid] * SIZE_HISTO;
        atomicAdd(&out[pos], 1u);
    }
}

bool StudentWorkImpl::isImplemented() const {
    return true;
}

void StudentWorkImpl::run_Histogram(
        OPP::CUDA::DeviceBuffer<float> &dev_value,
        OPP::CUDA::DeviceBuffer<unsigned> &dev_histogram,
        const unsigned width,
        const unsigned height
) {
    // TODO
    const unsigned size = width * height;
    dim3 threads((unsigned) SIZE_HISTO);
    dim3 blocks((size + threads.x - 1) / threads.x);
    /* clean histogram */
    unsigned zero_array[(unsigned) SIZE_HISTO] = {0};
    dev_histogram.copyFromHost(zero_array);

    computeHistogram<<<blocks, threads>>>(
            dev_histogram.getDevicePointer(),
            dev_value.getDevicePointer(),
            size);
}
