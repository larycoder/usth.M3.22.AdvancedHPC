#include "student.h"
#include <OPP_cuda.cuh>

namespace
{
    using namespace OPP::CUDA;
    
    /**
     * @brief Do your calculation here!
     */
    void computeDistance(DeviceBuffer<int> &d_distance, const int width)
    {
        // TODO: do the product ;-)
    }
}

/**
 * @brief Do not forget to return true!
 */
bool StudentWork1::isImplemented() const
{
    return false;
}

/**
 * @brief Computes the distance from any vertex to all the others one. 
 * The data are know thanks to incidence matrice.
 * The matrices are stored row per row, and are squared ones.
 *  
 * @param h_matrix the input matrix, stored row per row
 * @param h_studentMatrix the output matrix, row per row
 * @param width width (and height) of the square matrices
 */
void StudentWork1::computeDistance(int const *const h_matrix, int *const h_studentDistance, const int width)
{
    // copy the distance to GPU, do calculation, copy back to CPU
    ::DeviceBuffer<int> d_distance(h_matrix, width * width);
    ::computeDistance(d_distance, width);
    d_distance.copyToHost(h_studentDistance);
}
