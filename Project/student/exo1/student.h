#pragma once
#include <StudentWork.h>
#include <helper_math.h>

class StudentWork1 : public StudentWork
{
public:
	StudentWork1() = default; 
	StudentWork1(const StudentWork1&) = default;
	~StudentWork1() = default;
	StudentWork1& operator=(const StudentWork1&) = default;

	// todo
	bool isImplemented() const;
	void computeDistance( int const*const h_matrix, int *const h_studentDistance, const int width );
};