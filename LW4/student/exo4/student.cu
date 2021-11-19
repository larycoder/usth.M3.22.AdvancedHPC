#include <iostream>
#include <exo4/student.h>
#include <OPP_cuda.cuh>

namespace {
    struct PlusFunctor {
        __device__
        unsigned operator() (const unsigned &first, const unsigned &second) const {
            return first + second;
        }
    };

    struct HistoFunctor {
        __device__
        unsigned char operator() (const float &value) const {
            return value * 256.f;
        }
    };

    template<typename T, typename U>
    __global__
    void histogramEqualize(T *repartition, U *in, U *out, unsigned size) {
        unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size)
            out[tid] = (float(repartition[(unsigned char) (in[tid] * 256.f)]) * 255.f) / (size * 256.f);
    }
}

bool StudentWorkImpl::isImplemented() const {
    return true;
}

void StudentWorkImpl::run_Transformation(
        OPP::CUDA::DeviceBuffer<float> &dev_Value,
        OPP::CUDA::DeviceBuffer<unsigned> &dev_repartition,
        OPP::CUDA::DeviceBuffer<float> &dev_transformation // or "transformed"
) {
    // TODO
    OPP::CUDA::computeHistogram<float, unsigned, HistoFunctor>(
            dev_Value, dev_repartition,
            HistoFunctor());
    OPP::CUDA::inclusiveScan<unsigned, PlusFunctor>(
            dev_repartition, dev_repartition,
            PlusFunctor());

    dim3 threads(256);
    dim3 blocks((dev_Value.getNbElements() + 255) / 256);
    histogramEqualize<unsigned, float><<<blocks, threads>>>(
            dev_repartition.getDevicePointer(),
            dev_Value.getDevicePointer(),
            dev_transformation.getDevicePointer(),
            dev_transformation.getNbElements());
}
