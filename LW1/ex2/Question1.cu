#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <Exercise.hpp>
#include <include/chronoGPU.hpp>

namespace {
    /// function that returns the gather location for even/odd index
    class EvenOddFunctor : public thrust::unary_function<int, int> {
        long long m_size;
    public:
        EvenOddFunctor(unsigned m_size) : m_size(m_size) {}

        EvenOddFunctor() = delete;

        __device__ int operator()(const int &index) {
            if (index < (int) (m_size / 2)) {
                // left half array
                return (int) 2 * index;
            }
            // right half array
            return (int) 2 * (index - m_size / 2) + 1;
        }
    };
}

void Exercise::Question1(const thrust::host_vector<int> &A,
                         thrust::host_vector<int> &OE) const {
    ChronoGPU chrUp, chrDown, chrCalc;
    const long long size = static_cast<long long>(A.size());
    for (int i = 3; i--;) {
        chrUp.start();
        // TODO: move data up
        const thrust::device_vector<int> dA = A;
        thrust::device_vector<int> dOE(m_size);
        chrUp.stop();
        chrCalc.start();
        // TODO: calc
        thrust::counting_iterator<int> countIter(0);
        auto mapIter = thrust::make_transform_iterator(
                countIter,
                EvenOddFunctor(m_size));
        thrust::gather(
                mapIter, mapIter + m_size,
                dA.begin(),
                dOE.begin());
        chrCalc.stop();
        chrDown.start();
        // TODO: move data down
        OE = dOE;
        chrDown.stop();
    }
    std::cout << "Exercise 1 done in " << chrUp.elapsedTime() + chrDown.elapsedTime() + chrCalc.elapsedTime() << "ms:"
              << std::endl;
    std::cout << "- uptime  : " << chrUp.elapsedTime() << "ms." << std::endl;
    std::cout << "- gputime : " << chrCalc.elapsedTime() << "ms." << std::endl;
    std::cout << "- downtime: " << chrDown.elapsedTime() << "ms." << std::endl;
}
