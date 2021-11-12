cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_VERBOSE_MAKEFILE false)
set(PROJECT_NAME LW5)
set(TARGETS exo1 exo2 exo3 exo4)

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

include_directories(./src ./student ./utils)


# Need to put manually all the inputs, else MOC will fail :-(
add_executable(exo1
    src/exercise1/main.cu src/exercise1/ExerciseImpl.h src/exercise1/ExerciseImpl.cu
    src/reference/ImageEqualizer.h src/reference/ImageEqualizer.cpp
    student/exo1/student.h student/exo1/student.cu
    utils/chronoGPU.hpp utils/chronoGPU.cu utils/OPP.h utils/chronoCPU.cpp utils/chronoCPU.hpp
    utils/ppm.h utils/ppm.cpp
)
set_property(TARGET exo1 PROPERTY CUDA_ARCHITECTURES OFF)
add_executable(exo2
    src/exercise2/main.cu src/exercise2/ExerciseImpl.h src/exercise2/ExerciseImpl.cu
    src/reference/ImageEqualizer.h src/reference/ImageEqualizer.cpp
    student/exo2/student.h student/exo2/student.cu
    utils/chronoGPU.hpp utils/chronoGPU.cu utils/OPP.h utils/chronoCPU.cpp utils/chronoCPU.hpp
    utils/ppm.h utils/ppm.cpp
)
set_property(TARGET exo2 PROPERTY CUDA_ARCHITECTURES OFF)
add_executable(exo3
    src/exercise3/main.cu src/exercise3/ExerciseImpl.h src/exercise3/ExerciseImpl.cu
    src/reference/ImageEqualizer.h src/reference/ImageEqualizer.cpp
    student/exo3/student.h student/exo3/student.cu
    utils/chronoGPU.hpp utils/chronoGPU.cu utils/OPP.h utils/chronoCPU.cpp utils/chronoCPU.hpp
    utils/ppm.h utils/ppm.cpp
)
set_property(TARGET exo3 PROPERTY CUDA_ARCHITECTURES OFF)
add_executable(exo4
    src/exercise4/main.cu src/exercise4/ExerciseImpl.h src/exercise4/ExerciseImpl.cu
    src/reference/ImageEqualizer.h src/reference/ImageEqualizer.cpp
    student/exo4/student.h student/exo4/student.cu
    utils/chronoGPU.hpp utils/chronoGPU.cu utils/OPP.h utils/chronoCPU.cpp utils/chronoCPU.hpp
    utils/ppm.h utils/ppm.cpp
)
set_property(TARGET exo4 PROPERTY CUDA_ARCHITECTURES OFF)
