#include "b21.cuh"

// #define N 40000 // max 40000
#define IT 1000
#define NGPU 2

__global__ void JacobiIterationDistributed_v2(int n, float *a, float *x, int offset, int max,float* b, float*x_result){
    float buf = 0.0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x + offset; idx < max; idx += blockDim.x * gridDim.x){
        //printf("offset : %d, max: %d\n", offset, max);

        for (int idy = blockIdx.y * blockDim.y + threadIdx.y; idy < n; idy += blockDim.y * gridDim.y){
            if(idx != idy){
                buf += a[idx*n + idy] * x[idy];
            }
        }
        x_result[idx] = (b[idx] - buf)/a[idx + idx*n];
        //printf("idx: %d, sigma: %f, x_result: %f\n", idx,  buf, x_result[idx]);
    }
}



__global__ void JacobiIterationDistributed_v3(int n, float *a, float *x, int offset, int max, float* b, float*x_result){
    float buf = 0.0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x + offset; idx < max; idx += blockDim.x * gridDim.x){
        //printf("offset : %d, max: %d\n", offset, max);

        for (int idy = blockIdx.y * blockDim.y + threadIdx.y; idy < n; idy += blockDim.y * gridDim.y){
            if(idx != idy){
                buf += a[idx*n + idy] * x[idy];
            }
        }
        x_result[idx - offset] = (b[idx] - buf)/a[idx + idx*n];
        // printf("idx: %d, sigma: %f, x_result: %f\n", idx,  buf, x_result[idx - offset]);
    }
}
__global__ void mergeResults(int n, int nGPU, int offset, float *x_result, float *x){

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n/nGPU; idx += blockDim.x * gridDim.x){
        x[idx + offset] = x_result[idx];
        // printf("idx : %d, n/GPU: %d, offset: %d\n", idx, n/nGPU, offset);
        // printf("idx: %d, x_result: %f, x: %f\n", idx, x_result[idx], x[idx]);

    }
}

__global__ void initAMatrix(int n, float*a){
    long i;
    for (long j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x){
    
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
  
void Benchmark21::alloc(){
    cudaMallocManaged(&a_d, long(N)*long(N)*sizeof(float));
    cudaMallocManaged(&b_d, N*sizeof(float));
    cudaMallocManaged(&x_d, N*sizeof(float));
    cudaMallocManaged(&x_result_d, N*sizeof(float));

    s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * NGPU);
    for (int i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        err = cudaStreamCreate(&s[i]);
    }


    result_d = (float**)malloc(sizeof(float*)*NGPU);
    for (int i = 0; i < NGPU; i++) {
        cudaMallocManaged(&result_d[i], (N/NGPU)*sizeof(float));
    }
}
void Benchmark21::init(){

    for (int i = 0; i < N; i++ )
    {
        b_d[i] = 3.0;
        x_d[i] = 0.0;

        for ( int j = 0; j < N; j++ ){
            if ( i == j - 1 ){
                a_d[i+j*N] = -1.0;
            }
            else if ( j == i ){
                a_d[i+j*N] = 2.0;
            }
            else if ( j == i + 1 ){
                a_d[i+j*N] = -1.0;
            }
            else{
                a_d[i+j*N] = 0.0;
            }
        }
    }
    b_d[N-1] = ( float ) ( N + 1 );

}
void Benchmark21::reset(){
    for (int i = 0; i < N; i++ )
    {
        x_d[i] = 0.0;
    }
}
void Benchmark21::execute_sync(int iter){

    for ( int it = 0; it < IT; it++ ){  
        //printf("switch device\n");

        cudaSetDevice(0);
        cudaDeviceSynchronize();

        JacobiIterationDistributed_v2<<<1024, 32>>>(N, a_d, (it%2==0)?x_d:x_result_d, 0, N/NGPU, b_d, (it%2==0)?x_result_d:x_d);

        //printf("switch device\n");
        
        cudaSetDevice(1);
        cudaDeviceSynchronize();
        JacobiIterationDistributed_v2<<<1024, 32>>>(N, a_d, (it%2==0)?x_d:x_result_d, N/NGPU, N, b_d, (it%2==0)?x_result_d:x_d);
        
    }
}


void Benchmark21::execute_async(int iter){
    // int offset;
    // int section;
    // for ( int it = 0; it < IT; it++ ){  
    //     offset = 0;
    //     section = 0;
    //     for(int g = 0 ; g < NGPU; g++){
    //         offset = (N/NGPU) * g;
    //         section = (N/NGPU) * (g+1);
    //         cudaSetDevice(g);
    //         JacobiIterationDistributed_v2<<<1024, 32, 0, s[g]>>>(N, a_d, (it%2==0)?x_d:x_result_d, offset, section, b_d, (it%2==0)?x_result_d:x_d);
    //     }
    //     for (int j = 0; j < NGPU; j++) {
    //         err = cudaStreamSynchronize(s[j]);
    //     }

    // }

    int offset;
    int section;
    for ( int it = 0; it < IT; it++ ){  
        offset = 0;
        section = 0;
        for(int g = 0 ; g < NGPU; g++){
            offset = (N/NGPU) * g;
            section = (N/NGPU) * (g+1);
            cudaSetDevice(g);
            JacobiIterationDistributed_v3<<<1024, 32, 0,s[g]>>>(N, a_d, x_d, offset, section, b_d, result_d[g]);
            //printf("offset: %d, section: %d \n", offset, section);
        }

        for (int j = 0; j < NGPU; j++) {
            cudaSetDevice(j);
             
            err = cudaStreamSynchronize(s[j]);
            //cudaDeviceSynchronize();
        }

        cudaSetDevice(0);
        for (int j = 0; j < NGPU; j++) {
            offset = (N/NGPU) * j;
            mergeResults<<<1024, 32, 0, s[0]>>>(N, NGPU, offset, result_d[j], x_d);
        }
        err = cudaStreamSynchronize(s[0]);
    }

    float* a = (float*)malloc(sizeof(float)*N*N);
    float* x = (float*)malloc(sizeof(float)*N);
    float* b = (float*)malloc(sizeof(float)*N);
    float* x_res = (float*)malloc(sizeof(float)*N);
    for(int i = 0; i<N; i++){
        x[i] = 0.0;
        b[i] = 3.0;
    }
    b[N-1] = ( float ) ( N + 1 );
    for(int i = 0; i<N*N;i++){
        a[i] = a_d[i];
    }


    for(int it = 0; it < IT; it++){
        for(int i = 0; i < N; i++){
            float sigma = 0;
            for(int j = 0; j<N; j++){
                if(j!=i){
                    sigma = sigma + a[i*N + j]*x[j];
                }
            }

            x_res[i] = (b[i]-sigma)/a[i*N + i];
        }

        for(int k = 0; k<N; k++){
            x[k] = x_res[k];
        }
    }

    for(int i = 0; i<N; i++){
        // if(x[i] != x_d[i])
            printf("x: %f, x_d: %f\n", x[i], x_d[i]);
    }

}




void Benchmark21::execute_cudagraph(int iter){}
void Benchmark21::execute_cudagraph_manual(int iter){}
void Benchmark21::execute_cudagraph_single(int iter){}
std::string Benchmark21::print_result(bool short_form){
    return std::to_string(x_result_d[0]);
}