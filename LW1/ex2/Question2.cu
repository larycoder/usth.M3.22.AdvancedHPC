#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <Exercise.hpp>
#include <include/chronoGPU.hpp>

namespace {
    /// function that returns the scatter destination for even/odd index
    // TODO
    class EvenOddFunctor : public thrust::unary_function<int, int> {
        long long m_size;
    public:
        EvenOddFunctor(long long m_size) : m_size(m_size) {}

        EvenOddFunctor() = delete;

        __device__ int operator()(const int &index) {
            if (index % 2 == 0 /* even position */) {
                return index / 2;
            } else /* odd position */ {
                return (index - 1) / 2 + m_size / 2;
            }
        }
    };
}

void Exercise::Question2(const thrust::host_vector<int> &A,
                         thrust::host_vector<int> &OE) const {
    ChronoGPU chrUp, chrDown, chrCalc;
    const long long size = static_cast<long long>(A.size());
    for (int i = 3; i--;) {
        chrUp.start();
        // TODO: move data up
        thrust::device_vector<int> dA = A;
        thrust::device_vector<int> dOE(m_size);
        chrUp.stop();
        chrCalc.start();
        // TODO: do calc
        auto mapper = thrust::make_transform_iterator(
                thrust::counting_iterator<int>(0),
                EvenOddFunctor(m_size));
        thrust::scatter(
                thrust::device,
                dA.begin(), dA.end(),
                mapper, dOE.begin());
        chrCalc.stop();
        chrDown.start();
        // TODO: move data down
        OE = dOE;
        chrDown.stop();
    }
    std::cout << "Exercise 2 done in: " << chrUp.elapsedTime() + chrDown.elapsedTime() + chrCalc.elapsedTime() << "ms"
              << std::endl;
    std::cout << "- uptime  : " << chrUp.elapsedTime() << "ms." << std::endl;
    std::cout << "- gputime : " << chrCalc.elapsedTime() << "ms." << std::endl;
    std::cout << "- downtime: " << chrDown.elapsedTime() << "ms." << std::endl;
}
