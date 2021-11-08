#include <Exercise.hpp>
#include <include/chronoGPU.hpp>
#include <thrust/device_vector.h>

namespace quest2 {
	/// function that adds two elements and return a new one
    class AddFunctor : public thrust::binary_function<int, int, int> {
    public:
        __host__ __device__ int operator()(const int &x, const int &y) {
            return x + y;
        }
    };
}


void Exercise::Question2(thrust::host_vector<int>&A) const 
{
  	// TODO
	ChronoGPU chrUp, chrDown, chrCalc;
	for(int i=3; i--; ) {
		chrUp.start();
		// TODO: Move Data to DEVICE
        thrust::device_vector<int> dA(m_size);
        thrust::constant_iterator<int> constIter(5);
        thrust::counting_iterator<int> countIter(0);
		chrUp.stop();
		chrCalc.start();
		// TODO: Do the MAP
        thrust::transform(
                constIter, constIter + m_size,
                countIter,
                dA.begin(),
                quest2::AddFunctor());
		chrCalc.stop();
		chrDown.start();
		// TODO: Move Data to HOST
        A = dA;
		chrDown.stop();
	}        
	std::cout <<"Exercise 2 done in: "<<chrUp.elapsedTime()+chrDown.elapsedTime()+chrCalc.elapsedTime()<<"ms"<<std::endl;
	std::cout <<"- uptime  : "<<chrUp.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- gputime : "<<chrCalc.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- downtime: "<<chrDown.elapsedTime()<<"ms."<<std::endl;
}
