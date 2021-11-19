#pragma once
#include <StudentWork.h>
#include <helper_math.h>
#include <exo3/SparseMatrix.h>
#include <OPP_cuda.cuh>

class StudentWork3 : public StudentWork
{
public:

	// First question: just call the constructor.
	// Student, you have to write the constructor ... 
	SparseMatrix buildSparseMatrix(
		const unsigned m,
		const unsigned n,
		const std::vector<SparseValue>& sparseValues
	) {
		return SparseMatrix(m, n, sparseValues);	
	}

	// Question 2
	OPP::CUDA::DeviceBuffer<double> doQuestion2(const SparseMatrix&matrix, const OPP::CUDA::DeviceBuffer<double>&vector) {
		return matrix * vector;
	}

	// Question 3
	SparseMatrix doQuestion3(const SparseMatrix&A, const SparseMatrix& B) {
		return A * B;
	}

	bool isImplemented() const;

	bool isQuestion1Implemented() const;
	bool isQuestion2Implemented() const;
	bool isQuestion3Implemented() const;

	StudentWork3() = default; 
	StudentWork3(const StudentWork3&) = default;
	~StudentWork3() = default;
	StudentWork3& operator=(const StudentWork3&) = default;
};