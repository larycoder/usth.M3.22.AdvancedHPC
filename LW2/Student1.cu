// Question 1
#include <D_Matrix.cuh>
#include <H_Matrix.cuh>

#include <thrust/transform.h>
#include <thrust/functional.h>

bool D_Matrix::Exo1IsDone() {
	return false;
}

// returns this times that ...
D_Matrix D_Matrix::operator+(const D_Matrix& that) const
{
  // do "d_val + that.d_val" 
  D_Matrix result(m_n);
  return result;
}