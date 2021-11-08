#include <Exercise.hpp>
#include <include/chronoGPU.hpp>
#include <thrust/device_vector.h>

namespace quest1 {
    /// function that adds two elements and return a new one
    class AddFunctor : public thrust::binary_function<int, int, int> {
    public:
        __host__ __device__ int operator()(const int &x, const int &y) {
            return x + y;
        }
    };
}

void Exercise::Question1(
        const thrust::host_vector<int> &A,
        const thrust::host_vector<int> &B,
        thrust::host_vector<int> &C
) const {
    // TODO: addition of two vectors using thrust
    ChronoGPU chrUp, chrDown, chrCalc;
    for (int i = 3; i--;) {
        chrUp.start();
        // TODO: move data to DEVICE
        thrust::device_vector<int> dA = A;
        thrust::device_vector<int> dB = B;
        chrUp.stop();
        chrCalc.start();
        // TODO: DO the map
        thrust::device_vector<int> dC(m_size);
        thrust::transform(
                dA.begin(), dA.end(),
                dB.begin(),
                dC.begin(),
                quest1::AddFunctor());
        chrCalc.stop();
        chrDown.start();
        // TODO: move data to HOST
        C = dC;
        chrDown.stop();
    }
    std::cout << "Exercise 1 done in " << chrUp.elapsedTime() + chrDown.elapsedTime() + chrCalc.elapsedTime() << "ms:"
              << std::endl;
    std::cout << "- uptime  : " << chrUp.elapsedTime() << "ms." << std::endl;
    std::cout << "- gputime : " << chrCalc.elapsedTime() << "ms." << std::endl;
    std::cout << "- downtime: " << chrDown.elapsedTime() << "ms." << std::endl;
}
