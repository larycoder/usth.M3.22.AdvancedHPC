#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <Exercise.hpp>
#include <include/chronoGPU.hpp>

namespace {
	/// function that returns the gather location for even/odd index
	
}

void Exercise::Question1(const thrust::host_vector<int>& A,
						 thrust::host_vector<int>& OE ) const
{   
	ChronoGPU chrUp, chrDown, chrCalc;
	const long long size = static_cast<long long>(A.size());
	for(int i=3; i--; ) {
		chrUp.start();
		// TODO: move data up
		chrUp.stop();
		chrCalc.start();
		// TODO: calc
		chrCalc.stop();
		chrDown.start();									
		// TODO: move data down
		chrDown.stop();
	}
	std::cout <<"Exercise 1 done in "<<chrUp.elapsedTime()+chrDown.elapsedTime()+chrCalc.elapsedTime()<<"ms:"<<std::endl;
	std::cout <<"- uptime  : "<<chrUp.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- gputime : "<<chrCalc.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- downtime: "<<chrDown.elapsedTime()<<"ms."<<std::endl;
}
