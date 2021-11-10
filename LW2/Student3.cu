// Question 3
#include <D_Matrix.cuh>
#include <H_Matrix.cuh>

#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace {
    class diffuseFunctor : public thrust::unary_function<int, int> {
        const int m_n;
        const int row;

    public:
        diffuseFunctor(int m_n, int row) : m_n(m_n), row(row) {}
        diffuseFunctor() = delete;
        __device__ int operator()(const int &in) {
            int col = in % m_n;
            return col + row * m_n;
        }
    };
}

bool D_Matrix::Exo3IsDone() {
    return true;
}

void D_Matrix::diffusion(const int line, D_Matrix &result) const {
    auto mapper = thrust::make_transform_iterator(
            thrust::counting_iterator<int>(0),
            diffuseFunctor(m_n, line));

    thrust::gather(
            mapper, mapper + m_n * m_n,
            d_val, result.d_val);

}