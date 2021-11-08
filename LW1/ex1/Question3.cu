#include <Exercise.hpp>
#include <include/chronoGPU.hpp>
#include <thrust/device_vector.h>

namespace {
	// function that adds three elements and returns a new one
	
}

void Exercise::Question3(
	const thrust::host_vector<int>& A,
	const thrust::host_vector<int>& B, 
	const thrust::host_vector<int>& C, 
	thrust::host_vector<int>&D
) const {
  	// TODO: D=A+B+C
	ChronoGPU chrUp, chrDown, chrCalc;
	for(int i=3; i--; ) {
		chrUp.start();
		// TODO: Move Data to Device
		chrUp.stop();
		chrCalc.start();
		// TODO: Do the MAP
		chrCalc.stop();
		chrDown.start();									
		// TODO: Move Data to Host
		chrDown.stop();
	}
	std::cout <<"Exercise 3 done in "<<chrUp.elapsedTime()+chrDown.elapsedTime()+chrCalc.elapsedTime()<<"ms:"<<std::endl;
	std::cout <<"- uptime  : "<<chrUp.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- gputime : "<<chrCalc.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- downtime: "<<chrDown.elapsedTime()<<"ms."<<std::endl;
}
