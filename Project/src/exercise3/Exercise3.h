#pragma once

#include <ExerciseGPU.h>
#include <exo3/student.h>
#include <vector>
#include <random>

class Exercise3 : public ExerciseGPU 
{
public:
    Exercise3(const std::string& name ) 
        : ExerciseGPU(name, new StudentWork3()), verbose(true),
            m(8), n(8), o(8), 
            generator(randomDevice()), sA(nullptr), sB(nullptr), 
            studentVector(8), sC(nullptr)
    {}

    Exercise3() = delete;
    Exercise3(const Exercise3&) = default;
    ~Exercise3() = default;
    Exercise3& operator= (const Exercise3&) = default;

    Exercise3& parseCommandLine(const int argc, const char**argv) ;
    
private:
    bool verbose;

    void run(const bool verbose);
    bool check();

    void buildData();
    std::vector<SparseValue> generateASparseMatrix(const unsigned m, const unsigned n);
    void generateAVector(const unsigned size, std::vector<double>& vector);
    void runQuestion1();
    void runQuestion2();
    void runQuestion3();

    bool checkQuestion1();
    bool checkQuestion2();
    bool checkQuestion3();

    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int);


    std::random_device randomDevice;
    std::mt19937 generator;
    // size of the matrices 
    unsigned m, n, o;
    // sparse values of first matrix
    std::vector<SparseValue> matrixA;
    SparseMatrix *sA;
    // student product Matrix times Vector
    std::vector<double> aVector;
    std::vector<double> studentVector;
    // second matrix for multiplies
    std::vector<SparseValue> matrixB;
    SparseMatrix *sB;
    SparseMatrix *sC;    
};