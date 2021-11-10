// Question 2
#include <D_Matrix.cuh>
#include <H_Matrix.cuh>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace {
    class transposeFunctor : public thrust::unary_function<int, int> {
        const int m_n;

    public:
        transposeFunctor(int m_n) : m_n(m_n) {}
        transposeFunctor() = delete;
        __device__ int operator()(const int &in) {
            int col = in / m_n, row = in % m_n;
            int tRow = col, tCol = row;
            return tRow + tCol * m_n;
        }
    };
}

bool D_Matrix::Exo2IsDone() {
    return true;
}

// define the Matrix::transpose function
D_Matrix D_Matrix::transpose() const {
    D_Matrix result(m_n);
    auto mapper = thrust::make_transform_iterator(
            thrust::counting_iterator<int>(0),
            transposeFunctor(m_n));
    thrust::scatter(
            d_val, d_val + m_n * m_n,
            mapper,
            result.d_val);
    return result;
}