#include <exo2/student.h>
#include <OPP_cuda.cuh>

#define TID (threadIdx.x + blockIdx.x * blockDim.x)
#define THREADS 512
#define BLOCKS(size) ((size + THREADS - 1) / THREADS)
#define MEM (sizeof(int) + 0)

namespace {
    // add what you need here ...
    using namespace OPP::CUDA;

    /**
     * @brief Utilities function to print a vector
     */
    template<typename T, typename U>
    void print(T const *const matrix, const U size) {
        if (size >= U(10))
            return;
        std::cout << "Matrix of size " << size << "x" << size << std::endl;
        for (auto row = U(0); row < size; ++row) {
            for (auto column = U(0); column < size; ++column) {
                if (column > U(0))
                    std::cout << ", ";
                std::cout << matrix[row * size + column];
            }
            std::cout << std::endl;
        }
    }

    /**
     * @brief Utilities function to print a vector
     */
    template<typename T, typename U>
    void print(DeviceBuffer <T> &matrix, const U size) {
        if (size >= U(10))
            return;
        T *const temp = new T[size * size];
        matrix.copyToHost(temp);
        ::print(temp, size);
        delete temp;
    }

    namespace SPP {
        namespace DEV {
            template<typename T>
            __global__
            void catchNonZeroRow(int *out, T *data, int index, int size) {
                int tid = TID;
                if (tid < size * size) {
                    int row = (tid / size), col = (tid % size);
                    if (row >= index && col == index && data[tid] != T(0))
                        atomicCAS(out, -1, row); /* do not care which row get last */
                }
            }

            template<typename T>
            __global__
            void swapRow(T *row1, T *row2, int size) { /* only launch number of thread equal to row size */
                int tid = TID;
                if (tid < size) {
                    T temp = row1[tid];
                    row1[tid] = row2[tid];
                    row2[tid] = temp;
                }
            }

            template<typename T>
            __global__
            void subtractRow(T *row, T *diagRow, T coefficient, int size) {
                int tid = TID;
                if (tid < size)
                    row[tid] = row[tid] + coefficient * diagRow[tid];
            }

            template<typename T>
            __global__
            void reduceRow(T *data, int index, int size) { /* only launch num(below row) threads */
                int tid = TID;
                if (tid < (size - index - 1)) {
                    // this thread present single row
                    int row = tid + index + 1;
                    int col = index;
                    T lead = data[row * size + col];
                    if (lead != T(0)) {
                        // reduce whole row
                        T coefficient = T(-1) * (lead / data[index * size + index]);
                        subtractRow<T><<<BLOCKS(size), THREADS>>>(
                                data + row * size,
                                data + index * size,
                                coefficient, size);
                    }
                }
            }

            template<typename T>
            __global__
            void collectDiag(T *out, T *data, int size) {
                int tid = TID;
                if (tid < size)
                    out[tid] = data[tid * size + tid];
            }
        }

        template<typename T>
        struct mul {
            __device__
            T operator()(const T &first, const T &second) const {
                return first * second;
            }
        };

        template<typename T>
        struct MemManage {
            void *pointer;
            int *row_num;

            MemManage(int size) {
                cudaMalloc(&pointer, MEM);
                row_num = (int *) pointer;
            }

            ~MemManage() { cudaFree(pointer); }

            int getRowNum() {
                int row;
                cudaMemcpy(&row, row_num, sizeof(int), cudaMemcpyDeviceToHost);
                return row;
            }

            void setRowNum(int value) {
                cudaMemcpy(row_num, &value, sizeof(int), cudaMemcpyHostToDevice);
            }
        };

        template<typename T>
        class GaussDet {

            DeviceBuffer <T> matrix;

            MemManage<T> mem;

            const int size;
            T det = T(-1);
            T d = T(1);

            int findPivot(int i) {
                mem.setRowNum(-1);
                DEV::catchNonZeroRow<T><<<BLOCKS(size * size), THREADS>>>(
                        mem.row_num, matrix.getDevicePointer(), i, size);
                return mem.getRowNum();
            }

            void swapRow(int row, int index) {
                if (row == index) return; /* same row */
                DEV::swapRow<T><<<BLOCKS(size), THREADS>>>(
                        matrix.getDevicePointer() + index * size,
                        matrix.getDevicePointer() + row * size,
                        size);
                d *= T(-1); // count effect of swap action
            }

            void reduceOtherRow(int index) {
                DEV::reduceRow<T><<<BLOCKS((size - index - 1)), THREADS>>>(
                        matrix.getDevicePointer(), index, size);
            }

        public:
            GaussDet(DeviceBuffer <T> &matrix, const int size) :
                    matrix(matrix), size(size), mem(MemManage<T>(size)) {}

            GaussDet() = delete;

            void pivotToEchelon() {
                int row;
                for (int i = 0; i < (size - 1); i++) {
                    row = findPivot(i);
                    if (row < 0) { /* det equal to zero */
                        det = T(0);
                        return;
                    }
                    swapRow(row, i);
                    reduceOtherRow(i);
                }
            }

            T calDiagDet() {
                if (det == T(0)) return T(0);
                DeviceBuffer <T> diag(size);
                DEV::collectDiag<T><<<BLOCKS(size), THREADS>>>(
                        diag.getDevicePointer(),
                        matrix.getDevicePointer(),
                        size);
                return reduce<T, mul<T>>(diag, mul<T>(), T(1)) * d;
            }

            void printMatrix() {
                ::print<T>(matrix, size);
            }
        };
    }

    /**
     * @brief Do your calculation here, using the device!
     * @param matrix value of the square matrix, row per row
     * @param size width and heigth of the square matrix
     * @return the computed determinant
     */
    double computeDeterminant(DeviceBuffer<double> &matrix, const int size) {
        // TODO
        SPP::GaussDet<double> gauss(matrix, size); // once times class - do not reuse it
        gauss.pivotToEchelon();
        gauss.printMatrix();
        return gauss.calDiagDet();
    }
}

/**
 * @brief Do not forget to return true!
 */
bool StudentWork2::isImplemented() const {
    return true;
}

/**
 * @brief Exercise 2: compute a determinant of a given matrix.
 * @param h_matrix a matrix onto the host, stored row per row
 * @param size the width and height of the square matrix
 * @return the determinant of <code>h_matrix</code>
 */
double
StudentWork2::computeDeterminant(double const *const h_matrix, const int size) {
    ::DeviceBuffer<double> d_matrix(h_matrix, size * size);
    ::print<double>(h_matrix, size);
    double det = ::computeDeterminant(d_matrix, size);
    //::print<double>(d_matrix, size);
    return det;
}
