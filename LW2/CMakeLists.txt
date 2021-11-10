cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_VERBOSE_MAKEFILE false)
set(PROJECT_NAME Labwork2)
set(TARGETS exercise1)

# set CUDA
set(CUDA_STANDARD 14)
set(CUDA_STANDARD_REQUIRED ON)
set(CUDA_64_BIT_DEVICE_CODE ON)
set(CUDA_VERBOSE_BUILD  ON)

#get info
cmake_host_system_information(RESULT MY_INFO QUERY NUMBER_OF_LOGICAL_CORES)
MESSAGE(INFO ": Number of logical cores: ${MY_INFO}")
cmake_host_system_information(RESULT MY_INFO QUERY PROCESSOR_DESCRIPTION)
MESSAGE(INFO ": processor description: ${MY_INFO}")
cmake_host_system_information(RESULT MY_INFO QUERY OS_PLATFORM)
MESSAGE(INFO ": OS platform: ${MY_INFO}")
cmake_host_system_information(RESULT SYSTEM_NAME QUERY OS_NAME)
MESSAGE(INFO ": System name: ${SYSTEM_NAME}")

# set C++
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add CXX/CUDA compiler
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

include_directories(. ./utils)

# Need to put manually all the inputs, else MOC will fail :-(
add_executable(exercise1
    main.cu
    Student1.cu Student2.cu Student3.cu Student4.cu Student5.cu
    H_Matrix.cuh H_Matrix.cu
    D_Matrix.cuh D_Matrix.cu    
    utils/chronoCPU.cpp
    utils/chronoCPU.hpp
    utils/chronoGPU.cu
    utils/chronoGPU.hpp
    utils/common.hpp
)
set_property(TARGET exercise1 PROPERTY CUDA_ARCHITECTURES OFF)

