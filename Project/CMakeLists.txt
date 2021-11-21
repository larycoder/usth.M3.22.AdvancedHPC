cmake_minimum_required(VERSION 3.10.0)
set(PROJECT_NAME Project)
set(TARGETS exercise1 exercise2 exercise3)
set(CMAKE_VERBOSE_MAKEFILE   OFF)

# set CUDA
set(CUDA_STANDARD 14)
set(CUDA_STANDARD_REQUIRED ON)
set(CUDA_64_BIT_DEVICE_CODE ON)
set(CUDA_VERBOSE_BUILD  ON)

# add CUDA compiler
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

include_directories(src student utils ${CUDA_SAMPLES_INC})
# add specifics to UNIX (linux)
IF(UNIX)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=compute_35 --gpu-code=compute_35")
ENDIF()

# Need to put manually all the inputs, else MOC will fail :-(
add_executable(exercise1 
    src/exercise1/main.cu utils/Exercise.h
    src/exercise1/Exercise1.cu src/exercise1/Exercise1.h
    utils/chronoGPU.hpp utils/chronoGPU.cu
    utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo1/student.h student/exo1/student.cu
)
set_property(TARGET exercise1 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET exercise1 PROPERTY CUDA_ARCHITECTURES OFF)

add_executable(exercise2 
    src/exercise2/main.cu utils/Exercise.h
    src/exercise2/Exercise2.cu src/exercise2/Exercise2.h
    utils/chronoGPU.hpp utils/chronoGPU.cu
    utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo2/student.h student/exo2/student.cu
)
set_property(TARGET exercise2 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET exercise2 PROPERTY CUDA_ARCHITECTURES OFF)

add_executable(exercise3 
    src/exercise3/main.cu utils/Exercise.h
    src/exercise3/Exercise3.cu src/exercise3/Exercise3.h
    utils/chronoGPU.hpp utils/chronoGPU.cu
    utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo3/student.h 
    student/exo3/student1.cu
    student/exo3/student2.cu
    student/exo3/student3.cu
)
set_property(TARGET exercise3 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET exercise3 PROPERTY CUDA_ARCHITECTURES OFF)
