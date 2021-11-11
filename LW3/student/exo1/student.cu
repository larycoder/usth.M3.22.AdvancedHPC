#include "student.h"


// ==========================================================================================
// Exercise 1

namespace {
	// Feel free to add any function you need, it is your file ;-)

}

bool StudentWork1::isImplemented() const {
	return false;
}

// you should do this method to return the blue objects contained in the input parameter
thrust::device_vector<ColoredObject> StudentWork1::compactBlue( const thrust::device_vector<ColoredObject>& d_input ) {
	// it should work on GPU ;-)
	// use FLAG set to 1 for BLUE objects, 0 else
	// then do a SCAN to count the number of BLUE object, and obtain their relative position (+1)
	// At least scatter them into the anwser array !
	return thrust::device_vector<ColoredObject>(d_input);
}
