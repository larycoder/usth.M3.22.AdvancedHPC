#!/bin/bash
# uncomment the following lines for ictlab servers ...
#export CUDACXX=/usr/local/cuda/bin/nvcc
TARGETS="exercise1"
if [ "$1" != "" ]; then
    TARGETS=$1
fi
echo '\e[1;32;41mTarget is ' ${TARGETS} '\e[0m'
if [ -d linux ]; then
    echo '\e[1;32;41mFolder "linux" already exists\e[0m'
else
    mkdir linux
fi
cd linux
cmake ..
for exo in ${TARGETS} 
    do
        echo '\e[1;32;41mCompiling ' ${exo} '\e[0m'
        cmake --build . --config Release --target ${exo}
    done
cd ..

