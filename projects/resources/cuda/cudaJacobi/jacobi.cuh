/*! jacobi.cuh
 * 
 * Functions for handling Jacobi iterations on CUDA device.
 * 
 * \author Matthew McGonagle 
 */

#pragma once

//! Type for pointer to function for computing boundary values.
typedef float (*harmonic)(float, float);

/*! Compute a single Jacboi iteration for a single point on the PDE discretization grid.
 *
 * Compute a single Jacobi iteration for harmonic function (and Dirichlet boundary conditions).
 *
 * \param dimX The x-dimension of the PDE discretization grid.
 * \param dimY The y-dimension of the PDE discretization grid.
 * \param in Pointer to input values of iteration.
 * \param out Pointer to output values of iteration. 
 */
__global__ 
void doJacobiIteration(int dimX, int dimY, float * in, float * out);

/*! Copy values array from host to CUDA device.
 * 
 * \param values Pointer to values array.
 * \param dimensions Holds the x and y dimensions of the PDE discretization grid.
 * \param in Will hold value pointing to location of flat input array allocated on CUDA device.
 * \param out Will hold value pointing to location of flat output array allocated on CUDA device. 
 */
__host__
void copyToDevice(float * values, const int dimensions[2], float ** in, float ** out);

/*! Fill in the boundary values for flat array of grid values.
 * 
 * \param values Pointer to flat array to hold values.
 * \param dimensions Holds x and y dimensions of the PDE discretization grid.
 * \param lowerLeft The lower left corner coordinates of the xy-rectangular domain.
 * \param upperRight The upper right corner coordinates of the xy-rectangular domain. 
 * \param f The function to give the boundary values.
 */
__host__
void setBoundaryValues(float * values, const int dimensions[2], const float lowerLeft[2], const float upperRight[2], harmonic f);

/*! Set the initial values for the iteration process (including boundary values).
 *
 * \param dimensions Holds x and y dimensions of the PDE discretization grid.
 * \param lowerLeft The lower left corner coordinates of the rectangular xy-domain.
 * \param upperRight The upper right corner coordinates of the rectangular xy-domain.
 * \param f The function for the boundary values.
 */
__host__
float * makeInitialValues(const int dimensions[2], const float lowerLeft[2], const float upperRight[2], harmonic f);

/*! Set the true values of the solution.
 *
 * \param dimensions Holds x and y dimensions of the PDE discretization grid.
 * \param lowerLeft The lower left corner coordinates of the rectangular xy-domain.
 * \param upperRight The upper right corner coordinates of the rectangular xy-domain.
 * \param f The true values function.
 */
__host__
float * makeTrueValues(const int dimensions[2], const float lowerLeft[2], const float upperRight[2], harmonic f);

/*! Get the errors between the approximate solution and the true values.
 * 
 * \param values The values of the approximation.
 * \param trueValues The values of the true harmonic function.
 * \param dimensions The xy-dimensions of the PDE discretization grid.
 */
__host__
float * getErrors(const float * values, const float * trueValues, const int dimensions[2]);

/*! Get the logarithm of the relative errors between the approximation and true harmonic function.
 * 
 * \param errors The non-relative errors.
 * \param trueValues The true values of the harmonic function.
 * \param dimensions The xy-dimensions of the PDE discretization grid.
 * \param cutOff We round up absolute trueValues that are smaller than the cutOff to the cutOff. We also
                 round up to cutOff to avoid taking the logarithm of 0.
 */
__host__
float * getRelativeErrors(const float * errors, const float * trueValues, const int dimensions[2], float cutOff = 0.00001);

/*! Get the average absolute error.
 * 
 * \param values The values of the approximation.
 * \param trueValues The values of the true harmonic function.
 * \param dimensions The xy-dimensions of the PDE discretization grid.
 */
__host__
float getAverageError(const float * values, const float * trueValues, const int dimensions[2]); //const int dimX, const int dimY );

/*! Print values to standard output. Best to use for small dimensions.
 *
 * \param dimensions The xy-dimensions of the PDE discretization grid.
 * \param values The values to print.  
 */
__host__ 
void printValues(const int dimensions[2], const float * values);

/*! Save values of flat array to file as a binary file.
 *
 * The values are saved as an array of triplets as an (x-coordinate, y-coordinate, value). This format
 * is to be compatible with GNU plot.
 *  
 * \param values The values to save to file.
 * \param dimensions The xy-dimensions of the PDE discretization grid.
 * \param  lowerLeft The lower left corner coordinates of the rectangular xy-domain.
 * \param upperRight The upper right corner coordinates of the rectangular xy-domain.
 * \param filename File name to save to. 
 */
__host__
void saveToFile(const float * values, const int dimensions[2], const float lowerLeft[2], const float upperRight[2],
                const char * filename);
