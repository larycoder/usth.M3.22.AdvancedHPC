// Question 4
#include <D_Matrix.cuh>
#include <H_Matrix.cuh>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>

namespace {
    class keyFunctor : public thrust::unary_function<int, int> {
        const int m_n;

    public:
        keyFunctor(int m_n) : m_n(m_n) {}
        keyFunctor() = delete;
        __device__ int operator()(const int &in) {
            return in / m_n;
        }
    };

    class productFunctor : public thrust::unary_function<int, int> {
        const int m_n;
        const int col;

    public:
        productFunctor(int m_n, int col) : m_n(m_n), col(col) {}
        productFunctor() = delete;
        __device__ int operator()(const int &in) {
            return col + in * m_n;
        }
    };
}

bool D_Matrix::Exo4IsDone() {
    return true;
}

// returns this times that ...
D_Matrix D_Matrix::product1(const D_Matrix &that) const {
    D_Matrix result(m_n);
    D_Matrix difM(m_n);
    D_Matrix temp(m_n);
    thrust::device_vector<int> col(m_n);
    thrust::device_vector<int> keyOut(m_n);

    D_Matrix tpThat = that.transpose();
    const int count = m_n * m_n;
    auto keyIter = thrust::make_transform_iterator(
            thrust::counting_iterator<int>(0),
            keyFunctor(m_n));

    for (int i = 0; i < m_n; i++) {
        // map diffusion to matrix with * op
        tpThat.diffusion(i, difM);
        thrust::transform(
                d_val, d_val + count,
                difM.d_val, temp.d_val,
                thrust::multiplies<int>());

        // reduce to col
        thrust::reduce_by_key(
                thrust::device,
                keyIter, keyIter + count,
                temp.d_val,
                keyOut.begin(), col.begin());

        // scatter col to result
        auto mapper = thrust::make_transform_iterator(
                thrust::counting_iterator<int>(0),
                productFunctor(m_n, i));
        thrust::scatter(
                col.begin(), col.end(),
                mapper,
                result.d_val);
    }
    return result;
}