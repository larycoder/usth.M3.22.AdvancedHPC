#include <Exercise.hpp>
#include <include/chronoGPU.hpp>
#include <thrust/device_vector.h>

namespace {
	/// function that adds two elements and return a new one
}

void Exercise::Question1(
	const thrust::host_vector<int>& A,
	const thrust::host_vector<int>& B, 
	thrust::host_vector<int>&C
) const {
	// TODO: addition of two vectors using thrust
	ChronoGPU chrUp, chrDown, chrCalc;
	for(int i=3; i--; ) {
		chrUp.start();
		// TODO: move data to DEVICE
		chrUp.stop();
		chrCalc.start();
		// TODO: DO the map
		chrCalc.stop();
		chrDown.start();									
		// TODO: move data to HOST
		chrDown.stop();
	}
	std::cout <<"Exercise 1 done in "<<chrUp.elapsedTime()+chrDown.elapsedTime()+chrCalc.elapsedTime()<<"ms:"<<std::endl;
	std::cout <<"- uptime  : "<<chrUp.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- gputime : "<<chrCalc.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- downtime: "<<chrDown.elapsedTime()<<"ms."<<std::endl;
}
