#pragma once

#include <OPP_cuda.cuh>
#include <vector>

/**
 * @brief Non-zero sparse matrix cell. 
 * It is immutable, so you cannot modify the value/row/column of the instance.
 */
struct SparseValue 
{
    /* since test use assigment operator, const attribute cause error */
    // do not worry, student will keep in mind that vector values are const ;-)
    double value; // immutable
    unsigned row; // immutable
    unsigned column; // immutable

    SparseValue() = delete;
    SparseValue(const SparseValue&) = default;
    SparseValue& operator=(const SparseValue&) = default;
    // this is the only valuable constructor
    SparseValue(double value, unsigned row, unsigned column) : value(value), row(row), column(column) {}
};

/**
 * @brief Represents a Sparse Matrix on device, using a mixing between COO and CSR representations.
 *
 * A sparse matrix contains only the non-zero values of the matrix. 
 * Its size is m\times n, so made of m rows and n columns. 
 * The data are on the Cuda device only ...
 * 
 */
class SparseMatrix
{

public:
    const unsigned m; // number of rows
    const unsigned n; // number of columns;

    // the COO representation (length is number of non zero values)
    OPP::CUDA::DeviceBuffer<double> values;
    OPP::CUDA::DeviceBuffer<unsigned> columns;
    OPP::CUDA::DeviceBuffer<unsigned> rows;
    // the CSR row lookup; length is m 1
    OPP::CUDA::DeviceBuffer<unsigned> lookup;

    SparseMatrix() = delete;
    ~SparseMatrix() = default;
    SparseMatrix(const SparseMatrix&sm) : 
        m(sm.m), n(sm.n), values(sm.values), columns(sm.columns), 
        rows(sm.rows), lookup(sm.lookup)
    {}
    SparseMatrix(const unsigned m, const unsigned n, const std::vector<SparseValue>&sparseValues ) 
        :   m(m), n(n), 
            values(static_cast<const unsigned int>(sparseValues.size())), 
            columns(static_cast<const unsigned int>(sparseValues.size())), 
            rows(static_cast<const unsigned int>(sparseValues.size())), 
            lookup(m+1)
    {
        buildSparseMatrix(sparseValues);
    }
    unsigned getWidth() const { 
        return n; 
    }
    unsigned getHeight() const { 
        return m; 
    }    
    // question 1 ...
    void buildSparseMatrix(const std::vector<SparseValue>&);
    // question 2 ...
    OPP::CUDA::DeviceBuffer<double> operator*(const OPP::CUDA::DeviceBuffer<double>&) const;
    // question 3
    SparseMatrix operator*(const SparseMatrix&) const;    
};