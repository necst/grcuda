#include "b21.cuh"

// #define N 40000 // max 40000
#define IT 1001
#define NGPU 1

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




// void swap(float* &a, float* &b){
//     float *temp = a;
//     a = b;
//     b = temp;
// }
  
void Benchmark21::alloc(){
    cudaMallocManaged(&a_d, long(N)*long(N)*sizeof(float));
    cudaMallocManaged(&b_d, N*sizeof(float));
    cudaMallocManaged(&x_d, N*sizeof(float));
    cudaMallocManaged(&x_result_d, N*sizeof(float));
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
    int offset;
    int section;
    for ( int it = 0; it < IT; it++ ){  
        offset = 0;
        section = 0;
        for(int g = 0 ; g < NGPU; g++){
            offset = (N/NGPU) * g;
            section = (N/NGPU) * (g+1);
            cudaSetDevice(g);
            cudaDeviceSynchronize();
            JacobiIterationDistributed_v2<<<1024, 32>>>(N, a_d, (it%2==0)?x_d:x_result_d, offset, section, b_d, (it%2==0)?x_result_d:x_d);
        }

    }
}
void Benchmark21::execute_cudagraph(int iter){}
void Benchmark21::execute_cudagraph_manual(int iter){}
void Benchmark21::execute_cudagraph_single(int iter){}
std::string Benchmark21::print_result(bool short_form){
    return std::to_string(x_result_d[0]);
}

// int main(){
//     cudaEvent_t start, stop;
//     float elapsed=0;


//     // alloc





//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start, 0);


//     for ( int it = 0; it < IT; it++ ){  
//         //printf("switch device\n");

//         cudaSetDevice(0);
//         cudaDeviceSynchronize();

//         JacobiIterationDistributed_v2<<<1024, 32>>>(N, a_d, (it%2==0)?x_d:x_result_d, 0, N/NGPU, b_d, (it%2==0)?x_result_d:x_d);

//         //printf("switch device\n");
        
//         cudaSetDevice(1);
//         cudaDeviceSynchronize();
//         JacobiIterationDistributed_v2<<<1024, 32>>>(N, a_d, (it%2==0)?x_d:x_result_d, N/NGPU, N, b_d, (it%2==0)?x_result_d:x_d);
        
//     }
//     cudaSetDevice(0);
//     swap(x_d, x_result_d);
//     cudaEventRecord(stop, 0);
//     cudaEventSynchronize (stop);

//     cudaEventElapsedTime(&elapsed, start, stop); 
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     printf("execution time: %f \n", elapsed);
    
//     printf("-------------------------------------------------------\n");
//     // for(int i = 0; i < N; i++){
//     //     if(x[i]!=x_d[i]){
//     //         printf("error \n");
//     //     }
//     //     //printf("x: %f, x_d: %f\n", x[i], x_d[i]);

//     // }
//     printf("\n");

//     return 0;
// }