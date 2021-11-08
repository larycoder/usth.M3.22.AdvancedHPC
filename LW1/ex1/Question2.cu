#include <Exercise.hpp>
#include <include/chronoGPU.hpp>
#include <thrust/device_vector.h>

namespace {
	/// function that adds two elements and return a new one	
}


void Exercise::Question2(thrust::host_vector<int>&A) const 
{
  	// TODO
	ChronoGPU chrUp, chrDown, chrCalc;
	for(int i=3; i--; ) {
		chrUp.start();
		// TODO: Move Data to DEVICE
		chrUp.stop();
		chrCalc.start();
		// TODO: Do the MAP
		chrCalc.stop();
		chrDown.start();
		// TODO: Move Data to HOST
		chrDown.stop();
	}        
	std::cout <<"Exercise 2 done in: "<<chrUp.elapsedTime()+chrDown.elapsedTime()+chrCalc.elapsedTime()<<"ms"<<std::endl;
	std::cout <<"- uptime  : "<<chrUp.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- gputime : "<<chrCalc.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- downtime: "<<chrDown.elapsedTime()<<"ms."<<std::endl;
}
