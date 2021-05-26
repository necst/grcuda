#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N 10
#define IT 2

__global__ void JacobiIteration(int n, float *a, float *b, float *x, float*x_result){
    float sigma = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        sigma = 0;
        printf("thread id: %d, x : %f\n", i,x[i]);
        for(int j = 0 ; j<n; j++){
            if(j!=i){
                sigma = sigma + a[i + j * n]*x[j];
                printf("sigma_value %f\n",sigma);
            }
        }
        x_result[i] = (b[i] - sigma)/a[i + i*n];
        printf("thread id: %d, x_res : %f\n", i,x_result[i]);
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
    int i;
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
    float *a, *b, *x, *x_result;

    // alloc
    cudaMallocManaged(&a, N*N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&x_result, N*sizeof(float));

    // init
    for (int i = 0; i < N; i++ )
    {
        b[i] = 0.0;
    }
    b[N-1] = ( float ) ( N + 1 );

    for ( int i = 0; i < N; i++ )
    {
      x[i] = 0.0;
    }

    initAMatrix<<<32, 32>>>(N, a);

    for ( int it = 0; it < IT; it++ ){        

        JacobiIteration<<<32, 32>>>(N, a, b, x, x_result);
        cudaDeviceSynchronize();
        //swap(x, x_result);
        copy<<<32,32>>>(N,x,x_result);
        cudaDeviceSynchronize();

    }
    cudaDeviceSynchronize();
    for(int i = 0; i<N; i++){
        printf("%f ",x[i]);
    }
}