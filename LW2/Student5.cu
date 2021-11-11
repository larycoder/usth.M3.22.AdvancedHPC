// Question 5
#include <D_Matrix.cuh>
#include <H_Matrix.cuh>

#include <thrust/transform.h>
#include <thrust/functional.h>
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

    class transformFunctor : public thrust::unary_function<int, int> {
        const thrust::device_ptr<int> ptr1;
        const thrust::device_ptr<int> ptr2;
        const int line, m_n;

    public:
        transformFunctor(
                const thrust::device_ptr<int> ptr1,
                const thrust::device_ptr<int> ptr2,
                const int line, const int m_n
        ) : ptr1(ptr1), ptr2(ptr2), line(line), m_n(m_n) {}

        transformFunctor() = delete;

        __device__ int operator()(const int &in) {
            return ptr1[in] * ptr2[in % m_n + line * m_n];
        }
    };
}

bool D_Matrix::Exo5IsDone() {
    return true;
}

// returns this times that ...
D_Matrix D_Matrix::product2(const D_Matrix &that) const {
    D_Matrix result(m_n);
    thrust::device_vector<int> keyOut(m_n);

    D_Matrix tpThat = that.transpose();
    const int count = m_n * m_n;
    auto keyIter = thrust::make_transform_iterator(
            thrust::counting_iterator<int>(0),
            keyFunctor(m_n));

    for (int i = 0; i < m_n; i++) {
        // map diffusion to matrix with * op
        auto transformIter = thrust::make_transform_iterator(
                thrust::counting_iterator<int>(0),
                transformFunctor(d_val, tpThat.d_val, i, m_n));

        // reduce to col
        thrust::reduce_by_key(
                thrust::device,
                keyIter, keyIter + count,
                transformIter,
                keyOut.begin(), result.d_val + i * m_n);
    }
    return result.transpose();
}