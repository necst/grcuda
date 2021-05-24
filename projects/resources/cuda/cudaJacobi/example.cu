/*! example.cu
 *
 * Example to compute Jacobi iterations for specific size of grid discretization.
 *
 * \author Matthew McGonagle
 */

#include <iostream>
#include "jacobi.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <string>
#include <stdlib.h>

//! Harmonic function of x and y is used to compute true values and the boundary values. 
__host__
float getHarmonic(float x, float y) 
{
    //! Real Part ((z - 0.5 - 0.5i)^5) = (x - 0.5)^5 - 10 (x - 0.5)^3 (y - 0.5)^2 + 5 (x - 0.5) (y - 0.5)^4.

    x -= 0.5;
    y -= 0.5;
    return pow(x, 5) - 10 * pow(x, 3) * pow(y, 2) + 5 * pow(x, 1) * pow(y, 4);
}

/*! Compute order of N^2 Jacobi iterations for harmonic solution on xy unit square for boundary values where
 *      we divide the square into an NxN grid; save the results to file. 
 * 
 *  The number N is sent to the executable as a string and as the first and only parameter. The default value
 *      is 20 if no parameter is given. Also we require N > 1. 
 */
int main(int argc, char * argv[]) 
{
    // First get the dimensions from command line arguments.

    int N = 20;

    int nIterations = 3 * N * N, // For good convergence the number of iterations is of the same order as gridsize.
        dimensions[2] = {N, N}, // The dimensions of the grid to approximate PDE (not the CUDA execution grid).
        nThreads = N / 10 + 1, // Number of CUDA threads per CUDA block dimension. 
        memSize = dimensions[0] * dimensions[1] * sizeof(float);
    const float lowerLeft[2] = {0, 0},  // Lower left coordinate of rectangular domain.
                upperRight[2] = {1, 1}; // Upper right coordinate of rectangular domain.
    //! We use flat arrays, because CUDA uses flat arrays.
    float * values, * trueValues, * in, * out, * errors, * relErrors;
    const dim3 blockSize( nThreads , nThreads), // The size of CUDA block of threads.
               gridSize( (dimensions[0] + nThreads - 1) / nThreads, (dimensions[1] + nThreads - 1) / nThreads);
               /*  The number of blocks in CUDA execution grid; make sure there is atleast enough 
                *  threads to have one for each point in our differential equation discretization grid.
                *  There be extra threads that are unnecessary.
                */

    std::cout << "Making initial values and true values" << std::endl;
    // Initial values includes boundary values.
    values = makeInitialValues( dimensions, lowerLeft, upperRight, & getHarmonic ); 

    // Find the true values of harmonic function using the boundary values function.
    trueValues = makeTrueValues( dimensions, lowerLeft, upperRight, & getHarmonic );

    std::cout << "Before Average Error = " 
              << getAverageError(values, trueValues, dimensions) //dimensions[0], dimensions[1]) 
              << std::endl;

    // Need to copy values from host to CUDA device.
    std::cout << "Copying to Device" << std::endl;
    try 
    {
        copyToDevice(values, dimensions, &in, &out);
    }
    catch( ... )
    {
        std::cout << "Exception happened while copying to device" << std::endl;
    }

    // At end of loop, output is inside pointer *in.

    std::cout << "Doing Jacobi Iterations" << std::endl;
    for( int i = 0; i < nIterations; i++)
    {
        // Call CUDA device kernel to a Jacobi iteration. 
        doJacobiIteration<<< gridSize, blockSize >>>(dimensions[0], dimensions[1], in, out);
        cudaDeviceSynchronize();
        if(cudaGetLastError() != cudaSuccess)
        {
            std::cout << "Error Launching Kernel" << std::endl;
            return 1;
        }
        std::swap(in, out);
    }

    // Get the result from the CUDA device.
    std::cout << "Copying result to values" << std::endl;
    if(cudaMemcpy( values, in, memSize, cudaMemcpyDeviceToHost ) != cudaSuccess) 
    {
        std::cout << "There was a problem retrieving the result from the device" << std::endl;
        return 1;    
    }

    // Now compute errors and save important data to file.

    std::cout << "Copying to file 'values.dat'" << std::endl;
    saveToFile( values, dimensions, lowerLeft, upperRight, "data/values.dat");

    std::cout << "Now getting errors" << std::endl;
    errors = getErrors(values, trueValues, dimensions);
    saveToFile( errors, dimensions, lowerLeft, upperRight, "data/errors.dat");
    std::cout << "After Average Error = " 
              << getAverageError(values, trueValues, dimensions)
              << std::endl;

    std::cout << "Now getting relative errors" << std::endl;
    relErrors = getRelativeErrors(errors, trueValues, dimensions);
    saveToFile( relErrors, dimensions, lowerLeft, upperRight, "data/log10RelErrors.dat");

    // Clean up memory.

    cudaFree(in); // First clean up on CUDA device.
    cudaFree(out);
    delete values; // Now clean up on host.
    delete errors;
    delete relErrors;
    delete trueValues;

    return 0;
}
