#pragma warning( disable : 4244 ) 

#include <iostream>
#include <exercise1/Exercise1.h>
#include <helper_string.h>
#include <random>
#include <algorithm>
#include <limits>
#include <vector>
#include <thread>

// ==========================================================================================
void Exercise1::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << " [-s=n] [--size=n] [-h] [--help]"<< std::endl
        << "\twhere n is the number of vertices into the graph." << std::endl
        << std::endl;
}

// ==========================================================================================
void Exercise1::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void Exercise1::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
Exercise1& Exercise1::parseCommandLine(const int argc, const char**argv) 
{
    displayHelpIfNeeded(argc, argv);
    if( checkCmdLineFlag(argc, argv, "s") ) {
        const int value = getCmdLineArgumentInt(argc, argv, "s");
        if( value > 1 ) 
            size = value;
    } 
    if( checkCmdLineFlag(argc, argv, "size") ) {
        const int value = getCmdLineArgumentInt(argc, argv, "size");
        if( value > 1 ) 
            size = value;
    } 
    return *this;
}

void Exercise1::run(const bool verbose) {    
    const int nbTry = 5;
    if( verbose )
        std::cout << std::endl 
            << "Compute distance of graph with " << size << " vertices (using " 
            << nbTry << " tries ..." << std::endl;
    // build a host vector containing the matrix data
    generateDistance();
    StudentWork1& worker = *reinterpret_cast<StudentWork1*>(student);
    execute_and_display_GPU_time(
        verbose,
        [&]() -> void { worker.computeDistance( h_matrix, h_studentDistance, size ); },
        5
    );
}

void Exercise1::generateDistance() 
{
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice()); 
    std::uniform_int_distribution<> distributionValue(1, 42);
    std::uniform_int_distribution<> distributionArcExistance(0,2); 

    h_matrix = new int[size*size];
    h_studentDistance = new int[size*size];
    h_distance = new int[size*size];
    for(auto row=0; row<size; ++row)
        for(auto column=0; column<size; ++column) 
        {
            auto existance = distributionArcExistance(generator);
            h_matrix[row*size + column] = ( existance != 0 ) ? 
                distributionValue(generator) : 
                INT_MAX;
        }
}

void Exercise1::computeDistance() 
{   
    // let me do some smelling code :-)
    std::cout << "Compute the distance onto the HOST ..." << std::endl;
    memcpy(h_distance, h_matrix, size*size*sizeof(int));
    int*const temp = new int[size*size];
    for(auto pow=1; pow<size; pow<<=1)
    {
        std::vector<std::thread> threads;
        for(int row=0; row<size; ++row) { 
            threads.emplace_back(
                std::thread([&](int r) -> void {
                    for(auto column=0; column<size; ++column) 
                    {
                        int distance = h_distance[r*size + column];
                        for(auto i=0; i<size; ++i)
                        {
                            int d1 = h_distance[r*size+i];
                            int d2 = h_distance[i*size+column];
                            if( d1 < INT_MAX && d2 < INT_MAX )
                                distance = std::min(distance, d1 + d2);
                        }
                        temp[r*size + column] = distance;
                    }
                },
                row
            ));
        }
        for(auto& th : threads)
            th.join();
        memcpy(h_distance, temp, size*size*sizeof(int));
    }
    delete temp;
}

bool Exercise1::check() {
    computeDistance();
    for(auto row=0; row<size; ++row)
        for(auto column=0; column<size; ++column) 
        {
            const auto offset = row*size + column;
            if( h_distance[offset] != h_studentDistance[offset] ) 
            {
                if( size <= 32 ) {
                    std::cout << "*************** bad result, for matrix:" << std::endl;
                    print(h_matrix);
                    std::cout << "*************** we expect:" << std::endl;
                    print(h_distance);
                    std::cout << "*************** but obtain:" << std::endl;
                    print(h_studentDistance);
                }
                return false;
            }
        }
    return true;
}

void Exercise1::print(int*matrix)
{
    for(auto row=0; row<size; ++row) 
    {
        for(auto column=0; column<size; ++column) 
        {
            if( column > 0)
                std::cout << ", ";
            int value = matrix[row*size + column];
            if( value < INT_MAX )
                std::cout << value;
            else 
            std::cout << "X";
        }
        std::cout << std::endl;
    }
}
