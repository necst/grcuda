// https://matthewmcgonagle.github.io/blog/2019/01/25/CUDAJacobi


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N 20
#define M 5000

typedef float (*harmonic)(float, float);

__host__
float getHarmonic(float x, float y) 
{
    //! Real Part ((z - 0.5 - 0.5i)^5) = (x - 0.5)^5 - 10 (x - 0.5)^3 (y - 0.5)^2 + 5 (x - 0.5) (y - 0.5)^4.

    x -= 0.5;
    y -= 0.5;
    return pow(x, 5) - 10 * pow(x, 3) * pow(y, 2) + 5 * pow(x, 1) * pow(y, 4);
}

__host__
void setBoundaryValues(float * values, const int dimensions[2], const float lowerLeft[2], const float upperRight[2], harmonic f)
{
    float stride[2], pos;
    int i, last[2] = {dimensions[0] - 1, dimensions[1] - 1};
    float * memPos1, * memPos2; 

    for (i = 0; i < 2; i++)
        stride[i] = (upperRight[i] - lowerLeft[i]) / last[i];

    // Fill in top and bottom.

    memPos1 = values;
    memPos2 = values + (dimensions[1]-1);
    for (i = 0, pos = lowerLeft[0]; i < dimensions[0]; i++, pos += stride[0], memPos1+=dimensions[1], memPos2+=dimensions[1])
    {
        *memPos1 = f(pos, lowerLeft[1]); 
        *memPos2 = f(pos, upperRight[1]);
    }

    // Fill in sides.

    memPos1 = values + 1;
    memPos2 = values + (dimensions[0] - 1) * dimensions[1] + 1;

    for (i = 0, pos = lowerLeft[1]+stride[1]; i < dimensions[0] - 2; i++, pos += stride[1], memPos1++ , memPos2++ )
    {
        *memPos1 = f(lowerLeft[0], pos);
        *memPos2 = f(upperRight[0], pos);
    }
}

__host__
float * makeInitialValues( const int dimensions[2], const float lowerLeft[2], const float upperRight[2], harmonic f )
{
    float * values = new float[dimensions[0] * dimensions[1]],
          * rowPos = values,
          * colPos; 

    // We don't do anything for boundary values yet.

    rowPos = values + dimensions[1];
    for (int i = 0; i < dimensions[0] - 2; i++, rowPos += dimensions[1])
    {
        colPos = rowPos + 1;
        for (int j = 0; j < dimensions[1] - 2; j++, colPos++)
            *colPos = 0;    
    }
    setBoundaryValues( values, dimensions, lowerLeft, upperRight, f );

    return values;
}

__host__
float * makeTrueValues(const int dimensions[2], const float lowerLeft[2], const float upperRight[2], harmonic f)
{
  float *values = new float[dimensions[0] * dimensions[1]],
        *rowPosition = values,
        *colPosition; 

  float stride[2] {(upperRight[0] - lowerLeft[0]) / static_cast<float>(dimensions[0] - 1),
                    (upperRight[1] - lowerLeft[1]) / static_cast<float>(dimensions[1] - 1) }; 

  int i, j;
  float x, y;

  for (i = 0, x = lowerLeft[0]; i < dimensions[0]; i++, x += stride[0], rowPosition += dimensions[1]) 
  {
      colPosition = rowPosition;
      for (j = 0, y = lowerLeft[1]; j < dimensions[1] ; j++, y += stride[1], colPosition++)
          *colPosition = f(x, y);    
  }
  return values;
}


__host__
float getAverageError(const float * values, const float * trueValues, const int dimensions[2]) //dimX, const int dimY )
{
    // Now get the average error.
        double error = 0; 
        int offset;
        for (int i = 0; i < dimensions[0]; i++)
        {
            offset = i * dimensions[1];
            for (int j = 0; j < dimensions[1]; j++, offset++)
            {
                error += abs(values[offset] - trueValues[offset]);
            }
        } 
    
        error /= dimensions[0] * dimensions[1];
        return static_cast<float>(error);
}


__global__
void doJacobiIteration(int dimX, int dimY, float * in, float * out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x,
              j = blockIdx.y * blockDim.y + threadIdx.y;
    const int offset = i * dimY + j;

    // Remember to do nothing for boundary values.

    if( i < 1 || i > dimX - 2 ) 
        return;

    if( j < 1 || j > dimY - 2 )
        return;

    out += offset; 
    in += offset;

    // Jacobi iteration for harmonic means the ouput is average of neighbor points in grid.

    *out = *(in - 1) * 0.25 +
           *(in + 1) * 0.25 + 
           *(in - dimY) * 0.25 + 
           *(in + dimY) * 0.25;
}

/* Compute order of N^2 Jacobi iterations for harmonic solution on xy unit square for boundary values where
*  we divide the square into an NxN grid;
* 
*/

int main(){
    int nIterations = 3 * N * N, // For good convergence the number of iterations is of the same order as gridsize.
    dimensions[2] = {N, N}, // The dimensions of the grid to approximate PDE (not the CUDA execution grid).
    nThreads = N / 10 + 1, // Number of CUDA threads per CUDA block dimension. 
    memSize = dimensions[0] * dimensions[1] * sizeof(float);
    const float lowerLeft[2] = {0, 0},  // Lower left coordinate of rectangular domain.
                upperRight[2] = {1, 1}; // Upper right coordinate of rectangular domain.

    float * values, * trueValues, * in, * out, * errors, * relErrors;
    const dim3 blockSize( nThreads , nThreads), gridSize( (dimensions[0] + nThreads - 1) / nThreads, (dimensions[1] + nThreads - 1) / nThreads);
/*  The number of blocks in CUDA execution grid; make sure there is atleast enough 
*  threads to have one for each point in our differential equation discretization grid.
*  There be extra threads that are unnecessary.
*/


    //alloc 
    cudaMallocManaged(&in, N*N*sizeof(float));
    cudaMallocManaged(&out, N*N*sizeof(float));
    // init
    values = makeInitialValues( dimensions, lowerLeft, upperRight, & getHarmonic ); 
    for(int i = 0; i<N*N; i++){
        in[i] = values[i];
        out[i] = values[i];
    }

    // Find the true values of harmonic function using the boundary values function.
    trueValues = makeTrueValues( dimensions, lowerLeft, upperRight, & getHarmonic );

    std::cout << "Before Average Error = " << getAverageError(values, trueValues, dimensions)<< std::endl;
    
        

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


    std::cout << "After Average Error = " 
    << getAverageError(in, trueValues, dimensions)
    << std::endl;



    return 0;
}