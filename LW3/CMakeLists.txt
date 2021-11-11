cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_VERBOSE_MAKEFILE false)
set(PROJECT_NAME Labwork2)
set(TARGETS exo1 exo2)

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

include_directories(./src ./src/utils ./student)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# OPENMP for linux/mac but not for windows
IF(NOT "${SYSTEM_NAME}" MATCHES "Windows")
    set (CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS} -march=native -fomit-frame-pointer -O2 -Wno-ignored-attributes -mno-vzeroupper -fno-tree-vectorize")
ELSE()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") # /O2
ENDIF()

# Need to put manually all the inputs, else MOC will fail :-(
add_executable(exo1 
    src/exercise1/main.cu src/utils/Exercise.h
    src/exercise1/Exercise1.cu src/exercise1/Exercise1.h
    src/utils/chronoGPU.hpp src/utils/chronoGPU.cu
    src/utils/chronoCPU.hpp src/utils/chronoCPU.cpp
    student/exo1/student.h student/exo1/student.cu
)
set_property(TARGET exo1 PROPERTY CUDA_ARCHITECTURES OFF)

add_executable(exo2 
    src/exercise2/main.cu src/utils/Exercise.h
    src/exercise2/Exercise2.cu src/exercise2/Exercise2.h
    src/utils/chronoGPU.hpp src/utils/chronoGPU.cu
    src/utils/chronoCPU.hpp src/utils/chronoCPU.cpp
    student/exo2/student.h student/exo2/student.cu
)
set_property(TARGET exo2 PROPERTY CUDA_ARCHITECTURES OFF)
