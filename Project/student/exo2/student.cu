#include <exo2/student.h>
#include <OPP_cuda.cuh>

namespace {
	// add what you need here ...
	using namespace OPP::CUDA;

	/**
	 * @brief Utilities function to print a vector
	 */
	template<typename T,typename U>
	void print(T const*const matrix, const U size)
	{
		if( size >= U(10) )
			return;
		std::cout<<"Matrix of size "<<size<<"x"<<size<<std::endl;
		for(auto row=U(0); row<size; ++row) 
    	{
			for(auto column=U(0); column<size; ++column) 
			{
				if( column > U(0))
					std::cout << ", ";
				std::cout << matrix[row*size + column];
			}
			std::cout << std::endl;
		}
	}

	/**
	 * @brief Utilities function to print a vector
	 */
	template<typename T,typename U>
	void print(DeviceBuffer<T>&matrix, const U size)
	{
		if( size >= U(10) )
			return;
    	T*const temp = new T[size*size];
		matrix.copyToHost(temp);
		::print(temp, size);
		delete temp;
	}

	/**
	 * @brief Do your calculation here, using the device!
	 * @param matrix value of the square matrix, row per row
	 * @param size width and heigth of the square matrix
	 * @return the computed determinant
	 */
	double computeDeterminant(DeviceBuffer<double>&matrix, const int size)
	{
		// TODO
		double det; // = ???
		return det;
	}
}

/**
 * @brief Do not forget to return true!
 */
bool StudentWork2::isImplemented() const {
	return false;
}

/**
 * @brief Exercise 2: compute a determinant of a given matrix.
 * @param h_matrix a matrix onto the host, stored row per row
 * @param size the width and height of the square matrix
 * @return the determinant of <code>h_matrix</code>
 */
double
StudentWork2::computeDeterminant(double const*const h_matrix, const int size)
{
	::DeviceBuffer<double> d_matrix(h_matrix, size*size);
	::print<double>(h_matrix, size);
	double det = ::computeDeterminant(d_matrix, size);
	::print<double>(d_matrix, size);
	return det;
}
