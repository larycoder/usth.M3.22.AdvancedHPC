#pragma warning( disable : 4244 ) 

#include <iostream>
#include <helper_cuda.h>
#include <helper_string.h>
#include <exercise1/Exercise1.h>


int main(int argc, const char**argv) 
{
    // find and start a device ...
    std::cout<<"Find the device ..." << std::endl;
    int bestDevice = findCudaDevice(argc, argv);
    checkCudaErrors( cudaSetDevice( bestDevice ) );

    // run exercise 1
    Exercise1("Exercise 1")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}
