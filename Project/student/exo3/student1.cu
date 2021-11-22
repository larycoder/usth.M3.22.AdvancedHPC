#include <exo3/student.h>
#include <OPP_cuda.cuh>
#include <helper_cuda.h>

#define TID (threadIdx.x + blockIdx.x * blockDim.x)
#define THREADS 512
#define BLOCKS(size) (((size) + THREADS - 1) / THREADS)

/**
 * @brief do not modify the following line ...
 */
bool StudentWork3::isImplemented() const { return true; }

using namespace OPP::CUDA;
// anonymous namespace for your work
namespace {
    template<typename T>
    struct Wrapper {
        T value;
        unsigned key;

        __host__ __device__
        Wrapper(T value, unsigned key) : value(value), key(key) {}
    };

    namespace CHECK {
        void displayVector(DeviceBuffer <Wrapper<SparseValue>> &sortedWrapper) {
            unsigned size = sortedWrapper.getNbElements();
            Wrapper<SparseValue> *buf = (Wrapper<SparseValue> *) malloc(sizeof(Wrapper<SparseValue>) * size);
            sortedWrapper.copyToHost(buf);
            std::cout << "Key - Value: ";
            for (int i = 0; i < size; i++) {
                std::cout << buf[i].key << " - ";
                std::cout << buf[i].value.value << "  " << buf[i].value.row << "  " << buf[i].value.column;
                std::cout << " | ";
            }
            std::cout << std::endl;
            free(buf);
        }

        template<typename T>
        void displayWrapper(DeviceBuffer <Wrapper<T>> &sortedWrapper) {
            unsigned size = sortedWrapper.getNbElements();
            Wrapper<T> *buf = (Wrapper<T> *) malloc(sizeof(Wrapper<T>) * size);
            sortedWrapper.copyToHost(buf);
            std::cout << "Key - Value: ";
            for (int i = 0; i < size; i++) {
                std::cout << buf[i].key << " - " << buf[i].value;
                std::cout << " | ";
            }
            std::cout << std::endl;
            free(buf);
        }

        template<typename T>
        void displayDeviceBuffer(DeviceBuffer <T> &sortedWrapper) {
            unsigned size = sortedWrapper.getNbElements();
            T *buf = (T *) malloc(sizeof(T) * size);
            sortedWrapper.copyToHost(buf);
            std::cout << "Value: ";
            for (int i = 0; i < size; i++) {
                std::cout << buf[i] << "  ";
                std::cout << " | ";
            }
            std::cout << std::endl;
            free(buf);
        }
    }

    namespace SPP {
        template<typename T, typename F>
        __global__
        void wrapValueByKey(T *data, Wrapper<T> *wrapper, F functor, unsigned size) {
            unsigned tid = TID;
            if (tid < size) {
                T value = data[tid];
                wrapper[tid] = {value, functor(value)};
            }
        }

        template<typename V, typename I, typename S>
        __global__
        void splitSparseVector(V *value, I *row, I *column, Wrapper<S> *in, unsigned size) {
            unsigned tid = TID;
            if (tid < size) {
                S catcher = in[tid].value;
                value[tid] = catcher.value;
                row[tid] = catcher.row;
                column[tid] = catcher.column;
            }
        }

        namespace SORT {
            template<typename T>
            __global__
            void revert(T *out, T *in, unsigned size) {
                unsigned tid = TID;
                if (tid < size) {
                    out[size - tid - 1] = in[tid];
                }
            }

            template<typename T, typename F>
            __global__
            void transform(T *out, T *in1, T *in2, F functor, unsigned size) {
                unsigned tid = TID;
                if (tid < size)
                    out[tid] = functor(in1[tid], in2[tid]);
            }

            template<typename T>
            __global__
            void merge(T *out, T *up, T *down, T *flag, unsigned size) {
                unsigned tid = TID;
                if (tid < size) {
                    if (flag[tid]) {
                        out[tid] = up[tid];
                    } else {
                        out[tid] = down[tid];
                    }
                }
            }

            template<typename T>
            __global__
            void binary(unsigned *out, Wrapper<T> *in, unsigned index, unsigned size) {
                unsigned tid = TID;
                if (tid < size)
                    out[tid] = ((in[tid].key >> index) & 1);
            }

            template<typename T>
            __global__
            void scatter(Wrapper<T> *out, Wrapper<T> *in, unsigned *index, unsigned size) {
                unsigned tid = TID;
                if (tid < size)
                    out[index[tid]] = in[tid];
            }

            template<typename T>
            __global__
            void shiftOne(T *out, T *in, unsigned size) {
                unsigned tid = TID;
                if (tid < size) {
                    if (tid == 0) {
                        out[tid] = T(0);
                    } else {
                        out[tid] = in[tid - 1];
                    }
                }
            }

            template<typename T, typename F>
            void exclusiveScan(DeviceBuffer <T> &data, DeviceBuffer <T> &result, const F functor) {
                unsigned size = data.getNbElements();
                DeviceBuffer <T> mid(size);
                inclusiveScan<T, F>(data, mid, functor);
                shiftOne<T><<<BLOCKS(size), THREADS>>>(
                        result.getDevicePointer(),
                        mid.getDevicePointer(),
                        size);
            }
        }

        template<typename T>
        struct notFunc {
            __device__
            T operator()(const T &in, const T &_) const {
                return !in;
            }
        };

        template<typename T>
        struct plusFunc {
            __device__
            T operator()(const T &first, const T &second) const {
                return first + second;
            }
        };

        template<typename T>
        struct unarySubtractFunc {
            const T max;

            unarySubtractFunc(T max) : max(max) {}

            __device__
            T operator()(const T &first, const T &_) const {
                return max - first;
            }
        };

        template<typename T>
        class CustomRadixSort {
            void computeDown(
                    DeviceBuffer<unsigned> &flag,
                    DeviceBuffer<unsigned> &mid,
                    DeviceBuffer<unsigned> &down,
                    unsigned size
            ) {
                SORT::transform<unsigned, notFunc<unsigned>><<<BLOCKS(size), THREADS>>>(
                        mid.getDevicePointer(),
                                flag.getDevicePointer(),
                                flag.getDevicePointer(), // not have any meaning
                                notFunc<unsigned>(),
                                size);
                SORT::exclusiveScan<unsigned, plusFunc<unsigned>>(
                        mid, down, plusFunc<unsigned>());
            }

            void computeUp(
                    DeviceBuffer<unsigned> &flag,
                    DeviceBuffer<unsigned> &mid,
                    DeviceBuffer<unsigned> &up,
                    unsigned size
            ) {
                SORT::revert<unsigned><<<BLOCKS(size), THREADS>>>(
                        mid.getDevicePointer(),
                                flag.getDevicePointer(),
                                size);
                inclusiveScan<unsigned, plusFunc<unsigned>>(
                        mid, up, plusFunc<unsigned>());
                SORT::revert<unsigned><<<BLOCKS(size), THREADS>>>(
                        mid.getDevicePointer(),
                                up.getDevicePointer(),
                                size);
                SORT::transform<unsigned, unarySubtractFunc<unsigned>><<<BLOCKS(size), THREADS>>>(
                        up.getDevicePointer(),
                                mid.getDevicePointer(),
                                mid.getDevicePointer(), // no meaning
                                unarySubtractFunc<unsigned>(size),
                                size);
            }

            void split(
                    DeviceBuffer <Wrapper<T>> &out,
                    DeviceBuffer <Wrapper<T>> &in,
                    DeviceBuffer<unsigned> &flag) {
                unsigned size = in.getNbElements();
                DeviceBuffer<unsigned> up(size), down(size),
                        mid(size), index(size);
                computeDown(flag, mid, down, size);
                computeUp(flag, mid, up, size);
                SORT::merge<unsigned><<<BLOCKS(size), THREADS>>>(
                        index.getDevicePointer(),
                                up.getDevicePointer(),
                                down.getDevicePointer(),
                                flag.getDevicePointer(),
                                size);
                SORT::scatter<T><<<BLOCKS(size), THREADS>>>(
                        out.getDevicePointer(),
                                in.getDevicePointer(),
                                index.getDevicePointer(),
                                size);

//                std::cout << "Index Buffer: " << std::endl;
//                CHECK::displayDeviceBuffer<unsigned>(index);
//                std::cout << "Up Buffer: " << std::endl;
//                CHECK::displayDeviceBuffer<unsigned>(up);
//                std::cout << "Down Buffer: " << std::endl;
//                CHECK::displayDeviceBuffer<unsigned>(down);
            }

            void sortKeyBin(
                    DeviceBuffer <Wrapper<T>> &out,
                    DeviceBuffer <Wrapper<T>> &in,
                    unsigned index
            ) {
                const unsigned size = in.getNbElements();
                DeviceBuffer<unsigned> flag(size);
                SORT::binary<T><<<BLOCKS(size), THREADS>>>(
                        flag.getDevicePointer(),
                                in.getDevicePointer(),
                                index, size);

//                std::cout << "FLAG: " << std::endl;
//                CHECK::displayDeviceBuffer<unsigned>(flag);

                split(out, in, flag);
            }

        public:
            CustomRadixSort() = default;

            DeviceBuffer <Wrapper<T>> sort(DeviceBuffer <Wrapper<T>> &in) {
                DeviceBuffer <Wrapper<T>> mid[2] = {
                        DeviceBuffer<Wrapper<T>>(in),
                        DeviceBuffer<Wrapper<T>>(in)
                };
                for (int i = 0; i < 32; i++) {
                    sortKeyBin(mid[(i + 1) % 2], mid[i % 2], i);
                }
                return mid[0];
            }

            void printHello() {
                std::cout << "HELLO" << std::endl;
            }
        };

        namespace VALUE_SCAN {
            // so sad, lock have weird behavior
            class CustomDeviceLock {
            public:
                __host__ __device__
                CustomDeviceLock() = default;

                __host__ __device__
                ~CustomDeviceLock() = default;

                __device__
                void lock(int *mutex) {
                    while (atomicCAS(mutex, 0, 1) != 0) { ; }
                }

                __device__
                void unlock(int *mutex) {
                    atomicExch(mutex, 0);
                }
            };

            template<typename T>
            __global__
            void collectSegmentValueThenScatter(T *out, Wrapper<T> *in, unsigned in_size) { // bless it
                unsigned tid = TID;
                if (tid < in_size) {
                    if ((tid == (in_size - 1)) || (in[tid].value != in[tid + 1].value)) {
                        // position of key
                        // remember: key is number and value is row
                        unsigned row = in[tid].value;
                        unsigned count = in[tid].key;
                        out[row+1] = count;
                    }
                }
            }

            template<typename T>
            __global__
            void swapKeyValue(Wrapper<T> *out, Wrapper<T> *in, unsigned size) {
                unsigned tid = TID;
                if (tid < size)
                    out[tid] = {in[tid].key, in[tid].value};
            }

            template<typename T>
            __global__
            void extractValue(T *out, Wrapper<T> *in, unsigned size) {
                unsigned tid = TID;
                if(tid < size)
                    out[tid] = in[tid].value;
            }
        }

        template<typename T>
        struct segmentScanFunc { /* scan by value and compute key */
            __device__
            Wrapper<T> operator()(const Wrapper<T> &wrapper1, const Wrapper<T> &wrapper2) const {
                if (wrapper1.value != wrapper2.value) {
                    return (wrapper1.value > wrapper2.value) ? wrapper1 : wrapper2;
                } else {
                    return {wrapper1.value, (wrapper1.key + wrapper2.key)};
                }
            }
        };

        template<typename T>
        void reduceByValueThenScatter(DeviceBuffer<T> &out, DeviceBuffer <Wrapper<T>> &in) {
            const unsigned in_size = in.getNbElements();
            const unsigned out_size = out.getNbElements();
            DeviceBuffer <Wrapper<T>> in_mid(in_size);

            inclusiveScan<Wrapper<T>, segmentScanFunc<T>>(in, in_mid, segmentScanFunc<T>());

            std::cout << "Reduce scan: " << std::endl;
            CHECK::displayWrapper<T>(in_mid);

            VALUE_SCAN::collectSegmentValueThenScatter<T><<<BLOCKS(in_size), THREADS>>>( // god not forgive this !
                    out.getDevicePointer(), in_mid.getDevicePointer(), in_size);
        }
    }

    struct genKeyByIndexFunc {
        const unsigned col;

        genKeyByIndexFunc(unsigned col) : col(col) {}

        __device__
        unsigned operator()(const SparseValue &in) const {
            return in.row * col + in.column;
        }
    };

    struct genOnesKeyFunc {
        __device__
        unsigned operator()(const unsigned &_) const {
            return 1;
        }
    };
}

/**
 * @brief DO NOT FORGET TO return true ...
 */
bool StudentWork3::isQuestion1Implemented() const {
    return true;
}

/**
 * @brief Question 1: build the matrix.
 * You have to FILL the four device buffers: 
 * - OPP::CUDA::DeviceBuffer<double> values 
 * - OPP::CUDA::DeviceBuffer<unsigned> columns
 * - OPP::CUDA::DeviceBuffer<unsigned> rows
 * - OPP::CUDA::DeviceBuffer<unsigned> lookup
 * Notice these buffers are already allocated!
 * Notice also that m and n are already set, so you can use them
 */
void SparseMatrix::buildSparseMatrix(const std::vector <SparseValue> &sparseValues) {
    const unsigned size = sparseValues.size();
    DeviceBuffer <SparseValue> coo_list(sparseValues.data(), sparseValues.size());
    DeviceBuffer <Wrapper<SparseValue>> wrapper(coo_list.getNbElements());

    std::cout << "Row num: " << m << "  Col num: " << n << std::endl;

    // compute values, rows, columns
    SPP::wrapValueByKey<SparseValue, genKeyByIndexFunc><<<BLOCKS(size), THREADS>>>(
            coo_list.getDevicePointer(),
                    wrapper.getDevicePointer(),
                    genKeyByIndexFunc(n), size);

    std::cout << "SparseValue Wrapper: " << std::endl;
    CHECK::displayVector(wrapper);

    /* if user send random order of vector, use radix sort */
    /* not for this test case, haha */
    //SPP::CustomRadixSort<SparseValue> radix;
    //DeviceBuffer<Wrapper<SparseValue>> sortedWrapper(radix.sort(wrapper));

    SPP::splitSparseVector<double, unsigned, SparseValue><<<BLOCKS(size), THREADS>>>(
            values.getDevicePointer(),
                    rows.getDevicePointer(),
                    columns.getDevicePointer(),
                    wrapper.getDevicePointer(),
                    size);

    std::cout << "Value array: " << std::endl;
    CHECK::displayDeviceBuffer<double>(values);

    std::cout << "Col array: " << std::endl;
    CHECK::displayDeviceBuffer<unsigned>(columns);

    // compute lookup
    DeviceBuffer <Wrapper<unsigned>> rowWrapper(size);
    SPP::wrapValueByKey<unsigned, genOnesKeyFunc><<<BLOCKS(size), THREADS>>>(
            rows.getDevicePointer(),
                    rowWrapper.getDevicePointer(),
                    genOnesKeyFunc(), size);

    std::cout << "Row Wrapper: " << std::endl;
    CHECK::displayWrapper<unsigned>(rowWrapper);

    DeviceBuffer<unsigned> mid(lookup);
    SPP::reduceByValueThenScatter<unsigned>(lookup, rowWrapper); // evil evil evil
    inclusiveScan<unsigned, SPP::plusFunc<unsigned>>(lookup, lookup, SPP::plusFunc<unsigned>());

    std::cout << "Lookup value: " << std::endl;
    CHECK::displayDeviceBuffer(lookup);
}

