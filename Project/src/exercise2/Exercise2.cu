#pragma warning( disable : 4244 ) 

#include <iostream>
#include <exercise2/Exercise2.h>
#include <random>

namespace {
        

	template<typename T>
	void print(T const*const matrix, const int size)
	{
		if( size >= 10 )
			return;
		for(auto row=0; row<size; ++row) 
    	{
			for(auto column=0; column<size; ++column) 
			{
				if( column > 0)
					std::cout << ", ";
				std::cout << matrix[row*size + column];
			}
			std::cout << std::endl;
		}
	}
}

// ==========================================================================================
void Exercise2::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << " [-s=n] [-h] [--help]"<< std::endl
        << "\twhere n is the width/height of the matrix," << std::endl
        << "\tand both -h and --help display this help." << std::endl
        << std::endl;
}

// ==========================================================================================
void Exercise2::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void Exercise2::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
Exercise2& Exercise2::parseCommandLine(const int argc, const char**argv) 
{
    displayHelpIfNeeded(argc, argv);
    if( checkCmdLineFlag(argc, argv, "s") ) {
        const int value = getCmdLineArgumentInt(argc, argv, "s");
        if( value > 0 )
            size = value;
    }
    return *this;
}

void Exercise2::run(const bool verbose) {    
    // build a host vector containing the reference
    buildNonSingularMatrix();
    // go for the student calculation
    const int nbTry = 2;
    if( verbose )
        std::cout << std::endl 
            << "Compute determinant on GPU (" << nbTry << "tries)." 
            << std::endl;
    execute_and_display_GPU_time(
        verbose,
        [&]()->void{
            StudentWork2& worker = *reinterpret_cast<StudentWork2*>(student);
            student_determinant = worker.computeDeterminant( h_matrix, size );
        },
        nbTry
    );

    if( size > 256 ) {
        std::cout << std::endl << "***********" << std::endl;
        std::cout << "Relaunch with only 256*256 values to avoid numerical errors..." << std::endl;
        delete h_matrix;
        size = 256;
        buildNonSingularMatrix();
        StudentWork2& worker = *reinterpret_cast<StudentWork2*>(student);
        student_determinant = worker.computeDeterminant( h_matrix, size );
    }

    delete h_matrix;
}

bool Exercise2::check() 
{
    const double threshold = 1e-3;
    const bool result = fabs(student_determinant - 1.0) < threshold;
    if(!result) {
        std::cout << "- bad result: expected 1.0, got " << student_determinant << "!" <<std::endl;
    }
    return result;
}

void Exercise2::buildNonSingularMatrix() 
{
    double*const A = buildUpperMatrix();
    //::print(A,size);
    double*const B = buildUpperMatrix();
    //::print(B,size);
    h_matrix = multiplyUpperMatrixPerTransposeOfUpperMatrix(A, B);
    //::print(h_matrix, size);
    delete A;
    delete B;
}

double* Exercise2::buildUpperMatrix()
{
    std::cout << "build upper matrix of size " << size << std::endl;
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice()); 
    const double epsilon = 1e-10;
    std::uniform_real_distribution<> distributionValues(1.0-epsilon, 1.0+epsilon);

    double*const A = new double[size*size];
    for(auto row=0; row<size; ++row)
    {
        auto rowPtr = &A[row*size];
        // below diagonal, only zeros
        for(auto column=0; column<row; ++column)
            rowPtr[column] = 0.0;
        // diagonal contains only 1
        rowPtr[row] = 1.0;
        // above diagonal, random positive values
        for(auto column=row+1; column<size; ++column)
            rowPtr[column] = distributionValues(generator);
    }
    return A;
}

double* Exercise2::multiplyUpperMatrixPerTransposeOfUpperMatrix(double const*const A, double const*const transposeB) 
{
    std::cout << "multiply upper matrix per transpose of upper matrix of size " << size << std::endl;
    double* result = new double[size*size];
    for(auto row=0; row<size; ++row) 
    {
        const auto start = row*size;
        for(auto column=0; column<size; ++column)
        {
            double sum = 0.0;
            for(auto k=row; k<size; ++k)
                sum += A[start + k] * transposeB[column*size + k];
            result[start + column] = sum;
        }
    }
    return result;
}