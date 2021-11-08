#include <Exercise.hpp>
#include <include/chronoGPU.hpp>
#include <thrust/device_vector.h>

namespace quest3 {
	// function that adds three elements and returns a new one
    typedef thrust::tuple<int, int, int> tupleInt3;
    class AddFunctor : public thrust::unary_function<tupleInt3, int> {
    public:
        __host__ __device__ int operator()(const tupleInt3 &tuple) {
            return thrust::get<0>(tuple) + thrust::get<1>(tuple) + thrust::get<2>(tuple);
        }
    };
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
        thrust::device_vector<int> dA = A;
        thrust::device_vector<int> dB = B;
        thrust::device_vector<int> dC = C;
        thrust::device_vector<int> dD(m_size);
		chrUp.stop();
		chrCalc.start();
		// TODO: Do the MAP
        thrust::transform(
                thrust::make_zip_iterator(thrust::make_tuple(dA.begin(), dB.begin(), dC.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(dA.end(), dB.end(), dC.end())),
                dD.begin(),
                quest3::AddFunctor());
		chrCalc.stop();
		chrDown.start();									
		// TODO: Move Data to Host
        D = dD;
		chrDown.stop();
	}
	std::cout <<"Exercise 3 done in "<<chrUp.elapsedTime()+chrDown.elapsedTime()+chrCalc.elapsedTime()<<"ms:"<<std::endl;
	std::cout <<"- uptime  : "<<chrUp.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- gputime : "<<chrCalc.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- downtime: "<<chrDown.elapsedTime()<<"ms."<<std::endl;
}
