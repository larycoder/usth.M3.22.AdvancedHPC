#pragma once
#include <StudentWork.h>
#include <exo1/student.h>
#include <helper_math.h>


class StudentWork2 : public StudentWork1
{
public:

	bool isImplemented() const ;

	StudentWork2() = default; 
	StudentWork2(const StudentWork2&) = default;
	~StudentWork2() = default;
	StudentWork2& operator=(const StudentWork2&) = default;

	double computeDeterminant(double const*const h_matrix, const int size);
};