// Question 2
#include <D_Matrix.cuh>
#include <H_Matrix.cuh>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

bool D_Matrix::Exo2IsDone() {
	return false;
}
// define the Matrix::transpose function
D_Matrix D_Matrix::transpose() const
{
	D_Matrix result(m_n);
	return result;
}