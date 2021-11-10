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

bool D_Matrix::Exo4IsDone() {
	return false;
}
// returns this times that ...
D_Matrix D_Matrix::product1(const D_Matrix& that) const
{	
	D_Matrix result(m_n);
	return result;
}