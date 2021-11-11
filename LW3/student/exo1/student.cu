#include "student.h"


// ==========================================================================================
// Exercise 1

namespace {
    // Feel free to add any function you need, it is your file ;-)
    struct flagFunctor : public thrust::unary_function<ColoredObject, int> {
        __device__
        int operator()(const ColoredObject &color) {
            return color == ColoredObject::Color::BLUE ? 1 : 0;
        }
    };
}

bool StudentWork1::isImplemented() const {
    return true;
}

// you should do this method to return the blue objects contained in the input parameter
thrust::device_vector <ColoredObject> StudentWork1::compactBlue(const thrust::device_vector <ColoredObject> &d_input) {
    // it should work on GPU ;-)
    // use FLAG set to 1 for BLUE objects, 0 else
    // then do a SCAN to count the number of BLUE object, and obtain their relative position (+1)
    // At least scatter them into the anwser array !
    thrust::device_vector<int> flag(d_input.size());
    thrust::transform(
            d_input.begin(), d_input.end(),
            flag.begin(),
            flagFunctor());

    thrust::device_vector<int> scanVal(d_input.size());
    thrust::inclusive_scan(
            flag.begin(), flag.end(),
            scanVal.begin(), thrust::plus<int>());

    thrust::device_vector<ColoredObject> result(scanVal[scanVal.size() - 1]);
    auto mapper = thrust::make_transform_iterator(
            scanVal.begin(),
            thrust::placeholders::_1 - 1);
    thrust::scatter_if(
            d_input.begin(), d_input.end(),
            mapper,
            flag.begin(),
            result.begin());

    return result;
}
