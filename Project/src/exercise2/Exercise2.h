#pragma once

#include <ExerciseGPU.h>
#include <exo2/student.h>

class Exercise2 : public ExerciseGPU
{
public:
    Exercise2(const std::string& name ) 
        : ExerciseGPU(name, new StudentWork2()), size(8)
    {}

    Exercise2() = delete;
    Exercise2(const Exercise2&) = default;
    ~Exercise2() = default;
    Exercise2& operator= (const Exercise2&) = default;

    Exercise2& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);

    bool check();

    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int);

    int size;
    double student_determinant;
    double *h_matrix;

    void buildNonSingularMatrix();
    double* buildUpperMatrix();
    double* multiplyUpperMatrixPerTransposeOfUpperMatrix(double const*const, double const*const);
};