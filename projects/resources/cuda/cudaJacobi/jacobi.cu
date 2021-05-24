/*! jacobi.cu
 */

#include "jacobi.cuh"
#include <iostream>
#include <fstream>

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

__host__
void copyToDevice(float * values, const int dimensions[2], float ** in, float ** out)
{
    const int memSize = dimensions[0] * dimensions[1] * sizeof(float); 

    if (cudaMalloc( in, memSize ) != cudaSuccess)
        throw "Can't allocate in on device.";

    if (cudaMalloc( out, memSize ) != cudaSuccess)
        throw "Can't allocate out on device.";

    if(cudaMemcpy( *in, values, memSize, cudaMemcpyHostToDevice ) != cudaSuccess)
        throw "Can't copy values to in on device.";

    if(cudaMemcpy( *out, values, memSize, cudaMemcpyHostToDevice ) != cudaSuccess)
        throw "Can't copy values to out on device.";
 
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
float * getErrors(const float * values, const float * trueValues, const int dimensions[2])
{

    float * errors = new float[dimensions[0] * dimensions[1]];
    unsigned int position = 0;
    for ( int i = 0; i < dimensions[0]; i++)
    {
        for (int j = 0; j < dimensions[1]; j++, position++)
            errors[position] = values[position] - trueValues[position];
    }

    return errors;
}

__host__
float * getRelativeErrors(const float * errors, const float * trueValues, const int dimensions[2], float cutOff)
{
    float * relErrors = new float[dimensions[0] * dimensions[1]], * newError;

    float absError, absTrue;
    const float log10 = std::log(10);

    newError = relErrors;
    for(int i = 0; i < dimensions[0]; i++)
    {
        for(int j = 0; j < dimensions[1]; j++, newError++, errors++, trueValues++)
        {
            absError = abs(*errors);
            absTrue = abs(*trueValues);

            // Use a cutoff as a work around to dividing by 0.
            if (absTrue < cutOff)
                absTrue = cutOff;

            // Now use cutoff to work around logarithm of 0.
            if (absError / absTrue < cutOff)
                *newError = std::log(cutOff) / log10;
            else
                *newError = std::log(absError / absTrue) / log10; 
        }
    }  

    return relErrors;
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

__host__
void printValues(const int dimensions[2], const float * values) 
{
    const float * pos = values;
    for (int i = 0; i < dimensions[0]; i++) 
    {
        for (int j = 0; j < dimensions[1]; j++, pos++)
            std::cout << *pos << ",\t";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

__host__
void saveToFile(const float * values, const int dimensions[2], const float lowerLeft[2], const float upperRight[2],
const char * filename) 
{

   std::ofstream myFile(filename, std::ios::binary); 
   if(!myFile.is_open()) {
        throw "Unable to open file.";
   } 

   unsigned int sizeValues = dimensions[0] * dimensions[1] * sizeof(float);
   float * tuples = new float[dimensions[0] * dimensions[1] * 3], * coord;
   float position[2], skip[2];

   for(int i = 0; i < 2; i++)
   {
        position[i] = lowerLeft[i];
        skip[i] = (upperRight[i] - lowerLeft[i]) / (dimensions[i] - 1);
   }

   coord = tuples;
   for( int i = 0; i < dimensions[0]; i++, position[0] += skip[0])
   {
        position[1] = lowerLeft[1];
        for( int j = 0; j < dimensions[1]; j++, position[1] += skip[1], values++) 
        {
            *coord = position[0];
            coord++;
            *coord = position[1];
            coord++;
            *coord = *values; 
            coord++;
        }
   }

   myFile.write((const char *) tuples, 3 * sizeValues); 
   myFile.close();
   delete tuples;
}

