#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <exo2/student.h>
#include <bitset>

// Exercise 2: radix sort

namespace {
    // Add here what you need ...
    struct predicateFunctor : public thrust::unary_function<unsigned, int> {
        const int idx;
    public:
        predicateFunctor(int idx) : idx(idx) {}

        __device__
        int operator()(const unsigned &value) {
            return ((value >> idx) & 1);
        }
    };

    typedef thrust::tuple<int, int, int> tupleInt3;

    struct indexFunctor : public thrust::unary_function<tupleInt3, int> {
        __device__
        int operator()(const tupleInt3 &value) {
            auto flag = thrust::get<2>(value);
            auto iUp = thrust::get<0>(value);
            auto iDown = thrust::get<1>(value);
            if (flag) return iUp;
            return iDown;
        }
    };

    struct revertFunctor : public thrust::unary_function<int, int> {
        const int size;
    public:
        revertFunctor(int size) : size(size) {}

        __device__
        int operator()(const int &idx) {
            return size - 1 - idx;
        }
    };

    template<typename T>
    void display(thrust::device_vector <T> in) {
        thrust::host_vector <T> out = in;
        for (int i = out.size(); i--;) {
            std::bitset<8> x(out[i]);
            std::cout << x << " | ";
        }
        std::cout << std::endl;
    }
}

bool StudentWork2::isImplemented() const {
    return true;
}

thrust::device_vector<unsigned> StudentWork2::radixSortBase2(const thrust::device_vector<unsigned> &d_input) {
    const int size = d_input.size();

    thrust::device_vector<unsigned> result1 = d_input, result2(size);
    thrust::device_vector<int> iDown(size), iUp(size), revertFlag(size), index(size);

    thrust::device_vector<unsigned> *rPtr;
    thrust::device_vector<unsigned> *trPtr;

    auto count = thrust::counting_iterator<int>(0);
    auto revertIter = thrust::make_transform_iterator(
            count,
            revertFunctor(size));

    for (int i = 0; i < 32; i++) {
        // reference to result
        if (i % 2 == 0) {
            rPtr = &result1;
            trPtr = &result2;
        } else {
            rPtr = &result2;
            trPtr = &result1;
        }
        auto &result = *rPtr;
        auto &tempResult = *trPtr;

        // decide predicate key by bit i counting from right to left
        auto flag = thrust::make_transform_iterator(
                result.begin(),
                predicateFunctor(i));

        // index down
        auto completeFlag = thrust::make_transform_iterator(
                flag, 1 - thrust::placeholders::_1);
        thrust::exclusive_scan(
                completeFlag, completeFlag + size,
                iDown.begin(), 0);

        /* monitor result */
        //display(result);
        //display(iDown);
        //std::cout << "---------------" << std::endl;

        // index up
        thrust::scatter(
                flag, flag + size,
                revertIter,
                revertFlag.begin());
        thrust::inclusive_scan(
                revertFlag.begin(), revertFlag.end(),
                iUp.begin(),
                thrust::plus<unsigned>());
        thrust::scatter(
                iUp.begin(), iUp.end(),
                revertIter,
                revertFlag.begin());
        thrust::transform(
                revertFlag.begin(), revertFlag.end(),
                iUp.begin(),
                size - thrust::placeholders::_1);

        // final index
        thrust::transform(
                thrust::make_zip_iterator(thrust::make_tuple(iUp.begin(), iDown.begin(), flag)),
                thrust::make_zip_iterator(thrust::make_tuple(iUp.end(), iDown.end(), flag + size)),
                index.begin(),
                indexFunctor());

        // sort data by index
        thrust::scatter(
                result.begin(), result.end(),
                index.begin(),
                tempResult.begin());
    }

    /* monitor result */
    //display(result);

    return result1;
}

