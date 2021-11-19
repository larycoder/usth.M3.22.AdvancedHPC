#include <exo3/student.h>

namespace {
	
}


/**
 * @brief DO NOT FORGET TO return true ...
 */
bool StudentWork3::isQuestion2Implemented() const {
	return false;
}

/**
 * @brief Question 2: product between a matrix and a vector.
 * You must return the resulting vector as a DeviceBuffer<double> ...
 * @param V the left operand of the matrix*vector product
 * @return the result of the product this * V
 */
OPP::CUDA::DeviceBuffer<double> SparseMatrix::operator*(const OPP::CUDA::DeviceBuffer<double>& V) const
{
    OPP::CUDA::DeviceBuffer<double> result(m);
    return result;
}