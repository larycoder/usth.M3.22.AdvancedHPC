#include <iostream>
#include <exo2/student.h>
#include <OPP_cuda.cuh>

#define SIZE_HISTO 256.f

namespace {
    struct Functor {
        __device__
        unsigned operator () (const float &data ) const {
            return (unsigned) (data * SIZE_HISTO);
        }
    };
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
    OPP::CUDA::computeHistogram<float, unsigned, Functor>(
            dev_value, dev_histogram, Functor()
    );
}
