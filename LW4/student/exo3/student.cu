#include <iostream>
#include <exo3/student.h>
#include <OPP_cuda.cuh>

#define SIZE_HISTO 256
#define TID_CAL (threadIdx.x + blockIdx.x * blockDim.x)

namespace {
    namespace STUDENT_SCAN {
        using namespace OPP::CUDA;

        template<typename T, typename F>
        __host__
        void inclusiveScan(T *data, T *out, unsigned size, F functor);

        template<typename T>
        __device__
        void loadSharedMemory(T *data, unsigned size) {
            T *shared = getSharedMemory<T>();
            unsigned tid = TID_CAL;
            if (tid < size)
                shared[threadIdx.x] = data[tid];
            __syncthreads();
        }

        template<typename T>
        __device__
        void saveSharedMemory(T *data, unsigned size) {
            T *shared = getSharedMemory<T>();
            unsigned tid = TID_CAL;
            if (tid < size)
                data[tid] = shared[threadIdx.x];
            __syncthreads();
        }

        template<typename T, typename F>
        __device__
        void scanJumpingStep(unsigned jump, F functor, unsigned limit) {
            unsigned tid = threadIdx.x;
            T *shared = getSharedMemory<T>();
            T prevValue = shared[tid];
            __syncthreads();

            if (tid + jump < limit)
                shared[tid + jump] = functor(shared[tid + jump], prevValue);
            __syncthreads();
        }

        template<typename T, typename F>
        __global__
        void scanPerBlock(T *data, T *out, T *offset, unsigned size, F functor) {
            T *shared = getSharedMemory<T>();
            loadSharedMemory<T>(data, size);
            unsigned remain = size - blockDim.x * blockIdx.x;
            unsigned limit = (blockDim.x > remain ? remain : blockDim.x);
            for (int i = 1; i < blockDim.x; i <<= 1)
                scanJumpingStep<T, F>(i, functor, limit);
            saveSharedMemory<T>(out, size);
            if (threadIdx.x == 0)
                offset[blockIdx.x] = shared[blockDim.x - 1];
        }

        template<typename T, typename F>
        __global__
        void offsetCompute(T *data, T *offset, unsigned size, F functor) {
            unsigned tid = TID_CAL;
            if (tid < size)
                data[tid] = functor(data[tid], offset[blockIdx.x]);
        }

        template<typename T, typename F>
        __host__
        void scanStepOne(T *data, T *out, T *offset, unsigned size, F functor) {
            dim3 threads((unsigned) SIZE_HISTO);
            dim3 blocks((size + threads.x - 1) / threads.x);
            scanPerBlock<T, F><<<blocks, threads, threads.x * sizeof(T)>>>(data, out, offset, size, functor);
        }

        template<typename T, typename F>
        __host__
        void scanStepTwo(
                T *data, T *out, T *offset,
                unsigned data_size, unsigned offset_size,
                F functor) {
            if (offset_size > 1) {
                inclusiveScan<T, F>(offset, offset, offset_size, functor);
                dim3 threads((unsigned) SIZE_HISTO);
                dim3 blocks((data_size + threads.x - 1) / threads.x);
                offsetCompute<T, F><<<blocks, threads>>>(
                        data + threads.x, offset,
                        data_size - threads.x, functor);
            }
        }

        template<typename T, typename F>
        __host__
        void inclusiveScan(T *data, T *out, unsigned size, F functor) {
            const unsigned threads = SIZE_HISTO;
            const unsigned blocks = (size + threads - 1) / threads;
            T *offset;
            cudaMalloc(&offset, sizeof(T) * blocks);
            scanStepOne<T, F>(data, out, offset, size, functor);
            scanStepTwo<T, F>(data, out, offset, size, blocks, functor);
        }
    }

    struct plusFunctor {
        __device__
        unsigned operator()(const unsigned &first, const unsigned &second) {
            return first + second;
        }
    };
}

bool StudentWorkImpl::isImplemented() const {
    return true;
}

void StudentWorkImpl::run_Repartition(
        OPP::CUDA::DeviceBuffer<unsigned> &dev_histogram,
        OPP::CUDA::DeviceBuffer<unsigned> &dev_repartition
) {
    // TODO
    STUDENT_SCAN::inclusiveScan<unsigned, plusFunctor>(
            dev_histogram.getDevicePointer(),
            dev_repartition.getDevicePointer(),
            dev_histogram.getNbElements(),
            plusFunctor());
}
