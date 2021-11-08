#!/bin/bash
TARGETS="exercise1"
if [ "$1" != "" ]; then
    TARGETS=$1
fi
echo -e '\e[1;32;41mTarget is ' ${TARGETS} '\e[0m'
if [ -d linux ]; then
    echo -e '\e[1;32;41mFolder "linux" already exists\e[0m'
else
    mkdir linux
fi
cd linux
cmake ..
for exo in ${TARGETS} 
    do
        echo -e '\e[1;32;41mCompiling ' ${exo} '\e[0m'
        cmake --build . --config Release --target ${exo}
    done
cd ..

