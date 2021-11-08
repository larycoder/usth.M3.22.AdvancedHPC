cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_VERBOSE_MAKEFILE false)
set(PROJECT_NAME Labwork1)
set(TARGETS exercise2)

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

include_directories(. .. ../include ${CUDA_SAMPLES_INC})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# OPENMP for linux/mac but not for windows
IF(NOT "${SYSTEM_NAME}" MATCHES "Windows")
    set (CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS} -march=native -fomit-frame-pointer -O2 -Wno-ignored-attributes -mno-vzeroupper -fno-tree-vectorize")
ELSE()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") # /O2
ENDIF()

# Need to put manually all the inputs, else MOC will fail :-(
add_executable(exercise2
    Question1.cu Question2.cu Question3.cu Exercise.hpp main.cpp
    ../include/chronoCPU.cpp
    ../include/chronoCPU.hpp
    ../include/chronoGPU.cu
    ../include/chronoGPU.hpp
    ../include/common.hpp
)

set_property(TARGET exercise2 PROPERTY CUDA_ARCHITECTURES 52)
