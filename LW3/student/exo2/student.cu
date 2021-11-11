#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <exo2/student.h>

// Exercise 2: radix sort

namespace {
	// Add here what you need ...
}

bool StudentWork2::isImplemented() const {
	return false;
}

thrust::device_vector<unsigned> StudentWork2::radixSortBase2( const thrust::device_vector<unsigned>& d_input ) 
{	
	return thrust::device_vector<unsigned>(d_input);
}

