#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <Exercise.hpp>
#include <include/chronoGPU.hpp>

namespace {
	/// function that returns the scatter destination for even/odd index
	// TODO
}

template <typename T>
void Exercise::Question3(const thrust::host_vector<T>& A,
						thrust::host_vector<T>&OE) const 
{
  // TODO: D=A+B+C
	ChronoGPU chrUp, chrDown, chrCalc;
	const long long size = static_cast<long>(A.size());
	for(int i=3; i--; ) {
		chrUp.start();
		// TODO: move data up
		chrUp.stop();
		chrCalc.start();
		// TODO: do calc
		chrCalc.stop();
		chrDown.start();									
		// TODO: move data down
		chrDown.stop();
	}
	std::cout <<"Exercise 3 done in "<<chrUp.elapsedTime()+chrDown.elapsedTime()+chrCalc.elapsedTime()<<"ms:"<<std::endl;
	std::cout <<"- uptime  : "<<chrUp.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- gputime : "<<chrCalc.elapsedTime()<<"ms."<<std::endl;
	std::cout <<"- downtime: "<<chrDown.elapsedTime()<<"ms."<<std::endl;
}

// the structure ...
struct MyDataType {
	MyDataType(int i) : m_i(i) {}
	MyDataType() = default;
	MyDataType(const MyDataType&) = default;
	~MyDataType() = default;
	int m_i;
	operator int() const { return m_i; }
	// add what you want below ...
	double d=0.0;
	int array[5];
};


// Do not modify this method ;-)
void Exercise::checkQuestion3() const {
	const size_t size = sizeof(MyDataType)*m_size;
	std::cout<<"Check exercice 3 with arrays of size "<<(size>>20)<<" Mb"<<std::endl;
	checkQuestion3withDataType(MyDataType(0));
}
