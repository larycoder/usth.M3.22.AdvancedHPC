// Question 1
#include <D_Matrix.cuh>
#include <H_Matrix.cuh>

#include <thrust/transform.h>
#include <thrust/functional.h>

bool D_Matrix::Exo1IsDone() {
	return true;
}

// returns this times that ...
D_Matrix D_Matrix::operator+(const D_Matrix& that) const
{
  // do "d_val + that.d_val"
  D_Matrix result(m_n);
  thrust::transform(
          d_val, d_val + m_n*m_n,
          that.d_val,
          result.d_val,
          thrust::plus<int>());
  return result;
}