#include <exo3/student.h>

/**
 * @brief do not modify the following line ...
 */
bool StudentWork3::isImplemented() const { return true; }

// anonymous namespace for your work
namespace {
	
}

/**
 * @brief DO NOT FORGET TO return true ...
 */
bool StudentWork3::isQuestion1Implemented() const {
	return false;
}

/**
 * @brief Question 1: build the matrix.
 * You have to FILL the four device buffers: 
 * - OPP::CUDA::DeviceBuffer<double> values 
 * - OPP::CUDA::DeviceBuffer<unsigned> columns
 * - OPP::CUDA::DeviceBuffer<unsigned> rows
 * - OPP::CUDA::DeviceBuffer<unsigned> lookup
 * Notice these buffers are already allocated!
 * Notice also that m and n are already set, so you can use them
 */
void SparseMatrix::buildSparseMatrix(const std::vector<SparseValue>&sparseValues) 
{
}

