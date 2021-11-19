#pragma once

#include <ExerciseGPU.h>
#include <exo1/student.h>
#include <OPP/OPP_cuda_buffer.cuh>

class Exercise1 : public ExerciseGPU
{
public:
    Exercise1(const std::string& name ) 
        : ExerciseGPU(name, new StudentWork1()), size(32u)
    {}

    Exercise1() = delete;
    Exercise1(const Exercise1&) = default;
    ~Exercise1() = default;
    Exercise1& operator= (const Exercise1&) = default;

    Exercise1& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);

    bool check();
    
    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int);

    void generateDistance();
    void computeDistance();
    void print(int*matrix);

    int size; // size of the matrix (i.e. width and height)
    
    int *h_matrix;
    int *h_studentDistance;
    int *h_distance;
};