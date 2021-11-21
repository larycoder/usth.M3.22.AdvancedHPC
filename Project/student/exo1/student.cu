#include "student.h"
#include <OPP_cuda.cuh>
#include <helper_cuda.h>

#define TID_CAL (threadIdx.x + blockIdx.x * blockDim.x)
#define THREADS 512

namespace {
    /* Assumption that all matrix is square matrix */
    using namespace OPP::CUDA;

    namespace SPP { /* student parallel pattern */
        inline int BLOCKS(int size) { return (size + THREADS - 1) / THREADS; }

        namespace STUDENT_SCAN {
            template<typename T, typename K>
            struct DataKeyWrapper {
                T data;
                K key;

                __host__ __device__
                DataKeyWrapper(T data, K key) : data(data), key(key) {}
            };

            template<typename T>
            __global__
            void wrapDataByRow(T *data, DataKeyWrapper<T, unsigned> *wrapper, unsigned width) {
                unsigned tid = TID_CAL;
                unsigned size = width * width;
                if (tid < size)
                    wrapper[tid] = {data[tid], (tid / width)};
            }

            template<typename T, typename F>
            struct SegmentReduceFunctor {
                const F userFunctor;

                SegmentReduceFunctor(F f) : userFunctor(f) {}

                SegmentReduceFunctor() = delete;

                __device__
                DataKeyWrapper<T, unsigned> operator()(
                        const DataKeyWrapper<T, unsigned> &first,
                        const DataKeyWrapper<T, unsigned> &second
                ) const {
                    if (first.key != second.key) {
                        return (first.key > second.key) ? first : second;
                    } else {
                        T result = userFunctor(first.data, second.data);
                        return {result, first.key};
                    }
                }
            };

            template<typename T>
            __global__
            void collectRowScanResult(T *result, DataKeyWrapper<T, unsigned> *data, unsigned width) {
                unsigned tid = TID_CAL;
                unsigned size = width * width;
                if (tid < size) {
                    unsigned col = (tid % width), row = (tid / width);
                    auto d = data[tid];
                    if (d.key == row && col == (width - 1))
                        result[row] = d.data;
                }
            }
        }

        template<typename T, typename F>
        void reduce_by_row(T *data, T *result, unsigned width, F functor) {
            using namespace STUDENT_SCAN;
            const unsigned size = width * width;
            DeviceBuffer <DataKeyWrapper<T, unsigned>> wrapper(size);

            wrapDataByRow<T><<<BLOCKS(size), THREADS>>>(
                    data, wrapper.getDevicePointer(), width);
            getLastCudaError("Reduce by row go wrong in wrap step...");
            inclusiveScan<DataKeyWrapper<T, unsigned>, SegmentReduceFunctor<T, F>>(
                    wrapper, wrapper,
                    SegmentReduceFunctor<T, F>(functor));
            getLastCudaError("Reduce by row go wrong in scan step...");
            collectRowScanResult<T><<<BLOCKS(size), THREADS>>>(
                    result, wrapper.getDevicePointer(), width);
            getLastCudaError("Reduce by row go wrong in collect step...");
        }

        namespace COMMON {
            template<typename T>
            __global__
            void transpose(T *data, T *result, unsigned width) {
                unsigned tid = TID_CAL;
                unsigned size = width * width;
                if (tid < size) {
                    unsigned row = (tid / width), col = (tid % width);
                    result[row + col * width] = data[tid];
                }
            }

            template<typename T>
            __global__
            void diffuse(T *data, T *result, unsigned row, unsigned width) {
                unsigned tid = TID_CAL;
                unsigned size = width * width;
                if (tid < size) {
                    unsigned col = (tid % width);
                    result[tid] = data[row * width + col];
                }
            }

            template<typename T, typename F>
            __global__
            void transform(T *first, T *second, T *result, F functor, unsigned size) {
                unsigned tid = TID_CAL;
                if (tid < size)
                    result[tid] = functor(first[tid], second[tid]);
            }
        }

        template<typename T>
        struct plus {
            const T max;

            plus(T max) : max(max) {}

            __device__
            T operator()(const T &first, const T &second) const {
                T higher = (first > second) ? first : second;
                T lower = (first > second) ? second : first;
                if ((max - higher) <= lower)
                    return higher;
                return first + second;
            }
        };

        template<typename T>
        struct min {
            __device__
            T operator()(const T &first, const T &second) const {
                return (first > second) ? second : first;
            }
        };

        template<typename T>
        __host__
        void evilProduct(DeviceBuffer <T> &first, const int width) {
            using namespace COMMON;
            const unsigned size = width * width;

            DeviceBuffer <T> second(first),
                    secondTran(size),
                    middle(size),
                    result(size);

            transpose<T><<<BLOCKS(size), THREADS>>>(
                    second.getDevicePointer(),
                    secondTran.getDevicePointer(),
                    width);
            for (int i = 0; i < width; i++) {
                diffuse<T><<<BLOCKS(size), THREADS>>>(
                        secondTran.getDevicePointer(),
                        middle.getDevicePointer(),
                        i, width);
                transform<T, plus<T>><<<BLOCKS(size), THREADS>>>(
                        second.getDevicePointer(),
                        middle.getDevicePointer(),
                        middle.getDevicePointer(),
                        plus<T>(std::numeric_limits<T>::max()), size);
                reduce_by_row<T, min<T>>(
                        middle.getDevicePointer(),
                        result.getDevicePointer() + i * width,
                        width, min<T>());
            }
            transpose<T><<<BLOCKS(size), THREADS>>>(
                    result.getDevicePointer(),
                    middle.getDevicePointer(),
                    width);
            transform<T, min<T>><<<BLOCKS(size), THREADS>>>(
                    second.getDevicePointer(),
                    middle.getDevicePointer(),
                    first.getDevicePointer(),
                    min<T>(), size);
        }
    }

    /**
     * @brief Do your calculation here!
     */
    void computeDistance(DeviceBuffer<int> &d_distance, const int width) {
        // TODO: do the product ;-)
        unsigned count = log2(width) + 1;
        for (unsigned i = 0; i < count; i++) {
            SPP::evilProduct<int>(d_distance, width);
        }
    }
}

/**
 * @brief Do not forget to return true!
 */
bool StudentWork1::isImplemented() const {
    return true;
}

/**
 * @brief Computes the distance from any vertex to all the others one. 
 * The data are know thanks to incidence matrice.
 * The matrices are stored row per row, and are squared ones.
 *  
 * @param h_matrix the input matrix, stored row per row
 * @param h_studentMatrix the output matrix, row per row
 * @param width width (and height) of the square matrices
 */
void StudentWork1::computeDistance(int const *const h_matrix, int *const h_studentDistance, const int width) {
    // copy the distance to GPU, do calculation, copy back to CPU
    ::DeviceBuffer<int> d_distance(h_matrix, width * width);
    ::computeDistance(d_distance, width);
    d_distance.copyToHost(h_studentDistance);
}
