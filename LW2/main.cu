#include <cstdlib>
#include <iostream>
#include <ctime>
#include <algorithm>

#include <D_Matrix.cuh>
#include <H_Matrix.cuh>
#include <chronoCPU.hpp>
#include <chronoGPU.hpp>

#include <helper_cuda.h>
#include <helper_string.h>

namespace {

	void displayHelpIfNeeded(const int argc, const char**argv) 
	{
		if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
			std::cout << "Usage: " << argv[0] << " [-h] [--help] [-n=xxx]" << std::endl;
			std::cout << "\twhere options -h and --help display this help," << std::endl;
			std::cout << "\t  and option -n=xxx sets the width of the matrices to xxx" << std::endl;
			exit(0);
		}
	}

	int getNFromCmdLine(const int argc, const char**argv) 
	{
		const int minN = 1<<3;
		const int maxN = 1<<14;
		int N = 1<<8;
		if( checkCmdLineFlag(argc, argv, "n") ) {
			int value;
			getCmdLineArgumentValue(argc, argv, "n", &value);
			std::cout << "\tfind command line parameter -n=" << value << std::endl;
			if( value >= minN && value <= maxN )
				N = value;
			else
				std::cerr << "\tWarning: parameter must be greater to " << minN << " and lesser than " << maxN << std::endl;
		}
		return N;
	}

	void matrix_print(const H_Matrix& M) {
		for (int l = 0; l < M.m_n; ++l)
		{
			for (int c = 0; c < M.m_n; ++c)
				printf("% 4d ", M.h_val[l*M.m_n + c]);
			putchar('\n');
		}
	}

	bool test_equality(const H_Matrix& hM) 
	{
		return hM == hM;
	}

	bool test_addition(const H_Matrix& hA, const H_Matrix&hB, const D_Matrix& dA, const D_Matrix&dB) {
		// do the addition on GPU
		ChronoGPU gChr;
		gChr.start();
		const D_Matrix dResult = dA + dB;
		gChr.stop();

		// ground truth (CPU)
		ChronoCPU cChr;
		cChr.start();
		H_Matrix truth = hA + hB;
		cChr.stop();

		// print the timings
		std::cout << "==========================================" << std::endl;
		std::cout << "-> Matrices addition ..." << std::endl;
		std::cout << "Computation time on GPU:" << gChr.elapsedTime() << "ms." << std::endl;
		std::cout << "Computation time on CPU:" << cChr.elapsedTimeInMilliSeconds() << "ms." << std::endl;
		std::cout << "==========================================" << std::endl;

		H_Matrix hResult(hA.m_n);
		dResult.data(hResult.h_val);

		if (hA.m_n < 10) {
			std::cout << "Truth" << std::endl;
			matrix_print(truth);
			std::cout << "Device" << std::endl;
			matrix_print(hResult);
		}
		return truth == hResult;
	}

	bool test_transpose(const H_Matrix& hM, const D_Matrix& dM) 
	{
		// do the transpose on GPU
		ChronoGPU gChr;
		gChr.start();
		const D_Matrix dResult = dM.transpose();
		gChr.stop();

		// ground truth (CPU)
		ChronoCPU cChr;
		cChr.start();
		H_Matrix truth = hM.transpose();
		cChr.stop();

		// print the timings
		std::cout << "==========================================" << std::endl;
		std::cout << "-> Matrix transpose ..." << std::endl;
		std::cout << "Computation time on GPU:" << gChr.elapsedTime() << "ms." << std::endl;
		std::cout << "Computation time on CPU:" << cChr.elapsedTimeInMilliSeconds() << "ms." << std::endl;
		std::cout << "==========================================" << std::endl;

		H_Matrix hResult(hM.m_n);
		dResult.data(hResult.h_val);
		return truth == hResult;
	}


	bool test_diffusion(const H_Matrix& hA, const D_Matrix& dA)
	{
		// allocate the result
		D_Matrix diffused(hA.m_n);
		H_Matrix result(hA.m_n);
		H_Matrix truth(hA.m_n);
		// do it
		ChronoGPU gChr;
		ChronoCPU cChr;
		float gElapsed = 0.f, cElapsed = 0.f;
		int nb;
		const int nbLoops = (std::min)(128, hA.m_n);
		for (nb = 0; nb < nbLoops; ++nb) {
			gChr.start();
			dA.diffusion(nb, diffused);
			gChr.stop();
			gElapsed += gChr.elapsedTime();
			diffused.data(result.h_val);
			cChr.start();
			hA.diffusion(nb, truth);
			cChr.stop();
			cElapsed += cChr.elapsedTimeInMilliSeconds();
			if (result != truth)
				break;
		}	
		// print the timings
		std::cout << "==========================================" << std::endl;
		std::cout << "-> Diffusion of matrix (" << nb << " loops) ..." << std::endl;
		std::cout << "Computation time on GPU:" << (gElapsed/static_cast<float>(nb)) << "ms." << std::endl;
		std::cout << "Computation time on CPU:" << (cElapsed/static_cast<float>(nb)) << "ms." << std::endl;
		std::cout << "==========================================" << std::endl;

		return nb == nbLoops;
	}

	bool test_product(const H_Matrix& hA, const H_Matrix&hB, const D_Matrix& dA, const D_Matrix&dB) {
		// do the product on GPU
		ChronoGPU gChr;
		gChr.start();
		const D_Matrix dResult = dA.product1( dB );
		gChr.stop();
		
		// ground truth (CPU)
		ChronoCPU cChr;
		cChr.start();
		H_Matrix truth = hA * hB;
		cChr.stop();

		// print the timings
		std::cout << "==========================================" << std::endl;
		std::cout << "-> Matrices product ..." << std::endl;
		std::cout << "Computation time on GPU:" << gChr.elapsedTime() << "ms." << std::endl;
		std::cout << "Computation time on CPU:" << cChr.elapsedTimeInMilliSeconds() << "ms." << std::endl;
		std::cout << "==========================================" << std::endl;

		H_Matrix hResult(hA.m_n);
		dResult.data(hResult.h_val);

		if (hA.m_n < 10) {
			std::cout << "Truth" << std::endl;
			matrix_print(truth);
			std::cout << "Device" << std::endl;
			matrix_print(hResult);
		}
		return truth == hResult;
	}

	bool test_product_opt(const H_Matrix& hA, const H_Matrix&hB, const D_Matrix& dA, const D_Matrix&dB) {
		// do the product on GPU
		ChronoGPU gChr;
		gChr.start();
		const D_Matrix dResult = dA.product2(dB);
		gChr.stop();

		// ground truth (CPU)
		ChronoCPU cChr;
		cChr.start();
		H_Matrix truth = hA * hB;
		cChr.stop();

		// print the timings
		std::cout << "==========================================" << std::endl;
		std::cout << "-> Matrices product ..." << std::endl;
		std::cout << "Computation time on GPU:" << gChr.elapsedTime() << "ms." << std::endl;
		std::cout << "Computation time on CPU:" << cChr.elapsedTimeInMilliSeconds() << "ms." << std::endl;
		std::cout << "==========================================" << std::endl;

		H_Matrix hResult(hA.m_n);
		dResult.data(hResult.h_val);

		if (hA.m_n < 10) {
			std::cout << "Truth" << std::endl;
			matrix_print(truth);
			std::cout << "Device" << std::endl;
			matrix_print(hResult);
		}
		return truth == hResult;
	}

	void doExercice_1(
		const H_Matrix&hA, const H_Matrix&hB,
		const D_Matrix&dA, const D_Matrix&dB) 
	{
		if (D_Matrix::Exo1IsDone()) {
			// check the addition
			if (test_addition(hA, hB, dA, dB))
				std::cout << "--> Well done! Your ADDITION implementation seems to work." << std::endl;
			else
				std::cout << "--> Poor luck ... Your ADDITION implementation does not work." << std::endl;
		}
		else {
			std::cout << "Exercise 1 not implemented (D_Matrix::Exo1IsDone() returns false)" << std::endl;
		}
	}

	void doExercice_2(
		const H_Matrix&hA, const H_Matrix&hB,
		const D_Matrix&dA, const D_Matrix&dB) 
	{
		if (D_Matrix::Exo2IsDone()) {
			// check the transpose
			if (test_transpose(hB, dB))
				std::cout << "--> Well done! Your TRANSPOSE implementation seems to work." << std::endl;
			else
				std::cout << "--> Poor luck ... Your TRANSPOSE implementation does not work." << std::endl;
		}
		else {
			std::cout << "Exercise 2 not implemented (D_Matrix::Exo2IsDone() returns false)" << std::endl;
		}
	}
	

	void doExercice_3(
		const H_Matrix&hA, const H_Matrix&hB,
		const D_Matrix&dA, const D_Matrix&dB) 
	{
		// check the diffusion
		if (D_Matrix::Exo3IsDone()) {
			if (test_diffusion(hA, dA))
				std::cout << "--> Well done! Your DIFFUSION implementation seems to work." << std::endl;
			else
				std::cout << "--> Poor luck ... Your DIFFUSION implementation does not work." << std::endl; 
		}
		else {
			std::cout << "Exercise 3 not implemented (D_Matrix::Exo3IsDone() returns false)" << std::endl;
		}
	}
	

	void doExercice_4(
		const H_Matrix&hA, const H_Matrix&hB,
		const D_Matrix&dA, const D_Matrix&dB) 
	{
		// check the product
		if (D_Matrix::Exo4IsDone()) {
			if (test_product(hA, hB, dA, dB))
				std::cout << "--> Well done! Your PRODUCT implementation seems to work." << std::endl;
			else
				std::cout << "--> Poor luck ... Your PRODUCT implementation does not work." << std::endl;
		}
		else {
			std::cout << "Exercise 4 not implemented (D_Matrix::Exo4IsDone() returns false)" << std::endl;
		}
	}
	

	void doExercice_5(
		const H_Matrix&hA, const H_Matrix&hB,
		const D_Matrix&dA, const D_Matrix&dB) 
	{
		// check the advanced product
		if (D_Matrix::Exo5IsDone()) {
			if (test_product_opt(hA, hB, dA, dB))
				std::cout << "--> Well done! Your efficient PRODUCT implementation seems to work." << std::endl;
			else
				std::cout << "--> Poor luck ... Your efficient PRODUCT implementation does not work." << std::endl;
		}
		else {
			std::cout << "Exercise 5 not implemented (D_Matrix::Exo5IsDone() returns false)" << std::endl;
		}			
	}
	
	void doExercises(const int N) {
		srand(static_cast<unsigned int>(time(NULL)));

		H_Matrix hA( H_Matrix::random(N) );
		H_Matrix hB( H_Matrix::random(N) ); 
		
		D_Matrix dA( hA.export2Device() );
		D_Matrix dB( hB.export2Device() );
	
		// check that equality test works
		if (!test_equality(hA)) {
			std::cerr << "test equality of two matrices does not work ... cannot proceed!" << std::endl;
			exit(EXIT_FAILURE);
		}
		else
			std::cout << "test equality with success" << std::endl;
	
		// do the 5 exercises
		doExercice_1(hA, hB, dA, dB);
		doExercice_2(hA, hB, dA, dB);
		doExercice_3(hA, hB, dA, dB);
		doExercice_4(hA, hB, dA, dB);
		doExercice_5(hA, hB, dA, dB);
	}
}

int main(int ac, const char **av) {
	std::cout << "Labwork 2 : Matrix multiplication" << std::endl;
	// find and start a device ...
	std::cout<<"Find the device ..." << std::endl;
	int bestDevice = findCudaDevice(ac, av);
	checkCudaErrors( cudaSetDevice( bestDevice ) );
	
	// search parameters    
	displayHelpIfNeeded(ac, av);
	const int N = getNFromCmdLine(ac, av);
	std::cout << "Will use matrices of size " << N << "x" << N << "." << std::endl;
	std::cout << " -- a matrix needs " << (N*N*sizeof(int) >> 20) << " Mb." << std::endl;
	
	// do the exercises ...
	doExercises(N);

	return EXIT_SUCCESS;
}