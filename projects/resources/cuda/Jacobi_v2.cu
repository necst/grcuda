#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N 10000
#define IT 500

__global__ void JacobiIteration(int n, float *a, float *b, float *x, float*x_result){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        float sigma = 0;
        for(int j = 0 ; j<n; j++){
            if(j!=i){
                sigma += a[j + i * n]*x[j];
            }
        }
        x_result[i] = (b[i] - sigma)/a[i + i*n];
    }
}
__global__ void JacobiIterationDistributed(int n, float *a, float *x, float*sigma){
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n*n; tid += blockDim.x * gridDim.x){
        int i = tid/n;
        int j = tid%n;
        if(j!=i){
            atomicAdd(&sigma[i],a[tid]*x[j]);
        }
    }
}
__global__ void JacobiIterationDistributedResult(int n, float *a, float *b, float*x_result, float*sigma){
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x){
        x_result[tid] = (b[tid] - sigma[tid])/a[tid + tid*n];
        sigma[tid] = 0;
    }
}
__global__ void setSigmaToZero(int n, float*sigma){
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x){
       sigma[tid] = 0;
    }
}

__global__ void initAMatrix(int n, float*a){
    int i;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x){
    
        for ( i = 0; i < n; i++ ){
            if ( j == i - 1 ){
                a[j+i*n] = -1.0;
            }
            else if ( j == i ){
                a[j+i*n] = 2.0;
            }
            else if ( j == i + 1 ){
                a[j+i*n] = -1.0;
            }
            else{
                a[j+i*n] = 0.0;
            }
        }
    }
      
}

__global__ void copy(int n, float*a, float *b){
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x){
        a[j] = b[j];
    }
      
}

void swap(float* &a, float* &b){
    float *temp = a;
    a = b;
    b = temp;
  }
  


int main(){
    float *a, *b, *x, *x_result,*sigma;

    // alloc
    cudaMallocManaged(&a, N*N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&x_result, N*sizeof(float));
    cudaMallocManaged(&sigma, N*sizeof(float));
    // init

    for (int i = 0; i < N; i++ )
    {
        sigma[i] = 0;
    }

    for (int i = 0; i < N; i++ )
    {
        b[i] = 3.0;
    }
    b[N-1] = ( float ) ( N + 1 );

    for ( int i = 0; i < N; i++ )
    {
      x[i] = 0.0;
    }

    initAMatrix<<<32, 32>>>(N, a);

    float elapsed=0;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    for ( int it = 0; it < IT; it++ ){        
        JacobiIteration<<<32, 32>>>(N, a, b, x, x_result);
        cudaDeviceSynchronize();
        swap(x, x_result);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop);

    cudaEventElapsedTime(&elapsed, start, stop); 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("execution time: %f \n", elapsed);

    printf("-------------------------------------------------------\n");
    // for(int i = 0; i < N; i++){
    //     printf("%f ",x[i]);
    // }

    // printf("\n");

    for (int i = 0; i < N; i++ )
    {
        sigma[i] = 0;
    }

    for (int i = 0; i < N; i++ )
    {
        b[i] = 3.0;
    }
    b[N-1] = ( float ) ( N + 1 );

    for ( int i = 0; i < N; i++ )
    {
      x[i] = 0.0;
    }

    initAMatrix<<<32, 32>>>(N, a);


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    for ( int it = 0; it < IT; it++ ){  
        JacobiIterationDistributed<<<896, 32>>>(N, a, x, sigma);
        cudaDeviceSynchronize();
        JacobiIterationDistributedResult<<<896, 32>>>(N, a, b, x_result, sigma);
        cudaDeviceSynchronize();
        swap(x, x_result);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop );

    cudaEventElapsedTime(&elapsed, start, stop); 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("execution time: %f \n", elapsed);
    
    printf("-------------------------------------------------------\n");
    // for(int i = 0; i < N; i++){
    //     printf("%f ",x[i]);
    // }
    // printf("\n");

    return 0;
}