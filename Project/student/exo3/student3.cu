#include <exo3/student.h>

namespace {
	
}

/**
 * @brief DO NOT FORGET TO return true ...
 */
bool StudentWork3::isQuestion3Implemented() const {
	return false;
}



/**
 * @brief Question 3: product between a matrix and a matrix.
 * You must return the resulting matrix as a new SparseMatrix instance
 * @param that the left operand of the matrix*matrix product
 * @return the result of the product this * that
 */
SparseMatrix SparseMatrix::operator*(const SparseMatrix&that) const
{
    std::vector<SparseValue> sparseValues;
    return SparseMatrix(m, that.n, sparseValues);
}