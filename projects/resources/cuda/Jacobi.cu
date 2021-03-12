# include <cmath>
# include <cstdlib>
# include <ctime>
# include <iomanip>
# include <iostream>
# include <omp.h>
# include "Jacobi.cuh"


__global__ void Jacobi_update(float *x, float *b, float *xnew, int n){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        xnew[i] = b[i];
        if ( 0 < i )
        {
          xnew[i] = xnew[i] + x[i-1];
        }
        if ( i < n - 1 )
        {
          xnew[i] = xnew[i] + x[i+1];
        }
        xnew[i] = xnew[i] / 2.0;
    }
}

__global__ void reduction(float *x, float *xnew, int n, float d){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        d = d + pow ( x[i] - xnew[i], 2 );
    }
}

__global__ void reassign(float *x, float *xnew, int n){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      x[i] = xnew[i];
    }
}

void Jacobi::alloc(){
    err = cudaMallocManaged(&x, sizeof(float) * N);
    err = cudaMallocManaged(&b, sizeof(float) * N);
    err = cudaMallocManaged(&xnew, sizeof(float) * N);
}
void Jacobi::init(){
    for ( int i = 0; i < N; i++ )
    {
      b[i] = 0.0;
      x[i] = 0.0;
    }
    d = 0.0;
    b[N-1] = ( float ) ( N + 1 );
}
void Jacobi::reset(){
    for ( int i = 0; i < N; i++ )
    {
      b[i] = 0.0;
      x[i] = 0.0;
      xnew[i] = 0.0;
    }
    d = 0.0;

}
void Jacobi::execute_sync(int iter){
  for ( it = 0; it < iter; it++ )
  {
    Jacobi_update<<<num_blocks, block_size_1d>>>(x, b, xnew, N);
    reassign<<<num_blocks, block_size_1d>>>(x, xnew, N);
  }
}
void Jacobi::execute_async(int iter){
  printf("execute async \n");
}
void Jacobi::execute_cudagraph(int iter){}
void Jacobi::execute_cudagraph_manual(int iter){}
void Jacobi::execute_cudagraph_single(int iter){}
void Jacobi::prefetch(cudaStream_t &s1, cudaStream_t &s2){}
std::string Jacobi::print_result(bool short_form){}

