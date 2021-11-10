// Question 5
#include <D_Matrix.cuh>
#include <H_Matrix.cuh>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>

bool D_Matrix::Exo5IsDone() {
	return false;
}
// returns this times that ...
D_Matrix D_Matrix::product2(const D_Matrix& that) const {
	return D_Matrix(m_n);
}