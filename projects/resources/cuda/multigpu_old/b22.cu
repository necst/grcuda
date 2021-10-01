#include "b22.cuh"
#include <cstdio>
#include <time.h>

//#define NGPU 1
#define GPU0 0
#define GPU1 1
#define N_STREAMS 4 // must be even

#define A(i,j,N) A[(i)*N+(j)]
#define U(i,j,N) U[(i)*N+(j)]
#define L(i,j,N) L[(i)*N+(j)]

// Multi-kernel GPU implementation
// L matrix is transposed -> we obtain 2 upper triangular matrix
__global__ void updateU(const float *A, float *U, float *L, const int offset, const int max, const int dim, const int row){
	// int j = offset + blockIdx.x * blockDim.x + threadIdx.x;
	//int id = row * dim + j;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x + offset; j < max; j += blockDim.x * gridDim.x) {
		// if(j<max && j<dim && j >= row){
		if(j >= row){
			U(row,j,dim) = A(row,j,dim);
			for(int k = 0; k<row; k++)
				U(row,j,dim) -= L(k,row,dim)*U(k,j,dim);
		}
	}
}

__global__ void updateL(const float *A, float *U, float *L, const int offset, const int max, const int dim, const int col){
	// int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
	//int id = row * dim + j;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x + offset; i < max; i += blockDim.x * gridDim.x) {
		// if(i<max && i<dim && i>=col){
		if(i>=col){
			L(col,i,dim) = A(i,col,dim);
			for(int k = 0; k<col; k++)
				L(col,i,dim) -= L(k,i,dim)*U(k,col,dim);
			L(col,i,dim) /= U(col,col,dim);
		}
	}
}
  
void Benchmark22::alloc(){
	cudaMallocManaged(&A, long(N)*long(N)*sizeof(float));
	cudaMallocManaged(&U, long(N)*long(N)*sizeof(float));
	cudaMallocManaged(&L, long(N)*long(N)*sizeof(float));

	s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * NGPU);
	for (int i = 0; i < NGPU; i++) {
		cudaSetDevice(i);
		for (int j = 0; j < N_STREAMS; j++)
			err = cudaStreamCreate(&s[i*N_STREAMS+j]);
	}
}

void Benchmark22::init(){
	for(int i = 0; i<N; i++){
		for(int j = 0; j<N; j++){
			U(i,j,N) = 0.0;
			L(i,j,N) = 0.0;
			A(i,j,N) = float(rand()%100 + 1);
			if(rand()%2)
				A(i,j,N) *= -1;
		}
	}
}

void Benchmark22::reset(){
	srand(time(0));
	for(int i = 0; i<N; i++){
		for(int j = 0; j<N; j++){
			U(i,j,N) = 0.0;
			L(i,j,N) = 0.0;
			A(i,j,N) = float(rand()%100 + 1);
			if(rand()%2)
				A(i,j,N) *= -1;
		}
	}
}

void Benchmark22::execute_sync(int iter){
	int slice_GPU = N/NGPU+1;
	int slice_stream = slice_GPU/N_STREAMS+1;
	int offset,max;
	for (int it = 0; it < N; it++){
		for(int g = 0; g < NGPU; g++){
			cudaSetDevice(g);
			for (int i = 0; i < N_STREAMS; i++){
				offset = g*slice_GPU+i*slice_stream;
				max = (g*slice_GPU+(i+1)*slice_stream < (g+1)*slice_GPU) ? g*slice_GPU+(i+1)*slice_stream : (g+1)*slice_GPU;
				updateU<<<num_blocks,block_size_1d,0,s[g*N_STREAMS+i]>>>(A, U, L, offset, (max<N) ? max:N, N, it);
				err = cudaStreamSynchronize(s[g*N_STREAMS+i]);
			}
		}
		for(int g = 0; g < NGPU; g++){
			cudaSetDevice(g);
			for (int i = 0; i < N_STREAMS; i++){
				offset = g*slice_GPU+i*slice_stream;
				max = (g*slice_GPU+(i+1)*slice_stream < (g+1)*slice_GPU) ? g*slice_GPU+(i+1)*slice_stream : (g+1)*slice_GPU;
				updateL<<<num_blocks,block_size_1d,0,s[g*N_STREAMS+i]>>>(A, U, L, offset, (max<N) ? max:N, N, it);
				err = cudaStreamSynchronize(s[g*N_STREAMS+i]);
			}
		}	
	}
}

void Benchmark22::execute_async(int iter){

	// if (!pascalGpu || stream_attach) {
	// 	for (int i = 0; i < NGPU; i++) {
	// 		cudaStreamAttachMemAsync(s[i], A, long(N)*long(N)*sizeof(float));
	// 		cudaStreamAttachMemAsync(s[i], L, long(N)*long(N)*sizeof(float));
	// 		cudaStreamAttachMemAsync(s[i], U, long(N)*long(N)*sizeof(float));
	// 	}
    // }

	if (pascalGpu && do_prefetch) {
		for (int i = 0; i < NGPU; i++) {
			cudaMemPrefetchAsync(A, long(N)*long(N)*sizeof(float), i, s[i]);
			cudaMemPrefetchAsync(L, long(N)*long(N)*sizeof(float), i, s[i]);
			cudaMemPrefetchAsync(U, long(N)*long(N)*sizeof(float), i, s[i]);
		}
    }

	cudaEvent_t e1, e2[NGPU*(N_STREAMS/2)];
	int slice_GPU = N/NGPU+1;
	int slice_stream = slice_GPU/(N_STREAMS/2)+1;

	for (int it = 0; it < N; it++){
		for(int g = 0; g < NGPU; g++){		
			cudaSetDevice(g);
			for (int i = 0; i < N_STREAMS/2; i++){
				int offset = g*slice_GPU+i*slice_stream;
				int max = (g*slice_GPU+(i+1)*slice_stream < (g+1)*slice_GPU) ? g*slice_GPU+(i+1)*slice_stream : (g+1)*slice_GPU;
				updateU<<<num_blocks,block_size_1d,0,s[g*N_STREAMS+i]>>>(A, U, L, offset, (max<N) ? max:N, N, it);
				if(it/slice_stream == g*N_STREAMS+i){ 
					cudaEventCreate(&e1);
					cudaEventRecord(e1, s[g*N_STREAMS+i]);
				}
			}
		}
		for(int g = 0; g < NGPU; g++){
			for(int i =  N_STREAMS/2; i < N_STREAMS; i++)
				cudaStreamWaitEvent(s[g*N_STREAMS+i], e1, 0);
		}
        for(int g = 0; g < NGPU; g++){		
			cudaSetDevice(g);
			for (int i = 0; i < N_STREAMS/2; i++){
				int offset = g*slice_GPU+i*slice_stream;
				int max = (g*slice_GPU+(i+1)*slice_stream < (g+1)*slice_GPU) ? g*slice_GPU+(i+1)*slice_stream : (g+1)*slice_GPU;
				updateL<<<num_blocks,block_size_1d,0,s[g*N_STREAMS+i+N_STREAMS/2]>>>(A, U, L, offset, (max<N) ? max:N, N, it);
				// cudaEventCreate(&e2[g*(N_STREAMS/2)+i]);
				// cudaEventRecord(e2[g*(N_STREAMS/2)+i], s[g*N_STREAMS+i+N_STREAMS/2]);
			}
		}
		for(int g = 0; g < NGPU; g++){
			for(int i = 0; i < N_STREAMS/2; i++)
				err = cudaStreamSynchronize(s[g*N_STREAMS+i+N_STREAMS/2]);
				// cudaStreamWaitEvent(s[g*N_STREAMS+i], e2[g*(N_STREAMS/2)+i], 0);
		}
	}
	// for(int g = 0; g < NGPU; g++){
	// 	for(int i = 0; i < N_STREAMS/2; i++)
	// 		err = cudaStreamSynchronize(s[g*N_STREAMS+i+N_STREAMS/2]);
	// }
}

void Benchmark22::execute_cudagraph(int iter){}
void Benchmark22::execute_cudagraph_manual(int iter){}

void Benchmark22::execute_cudagraph_single(int iter){
	cudaSetDevice(0);
	for (int it = 0; it < N; it++){
		updateU<<<num_blocks,block_size_1d>>>(A, U, L, 0, N, N, it);
		cudaDeviceSynchronize();
		updateL<<<num_blocks,block_size_1d>>>(A, U, L, 0, N, N, it);
		cudaDeviceSynchronize();
	}
}

std::string Benchmark22::print_result(bool short_form){
	std::string res;
	float err = 0.0;
	for(int i = N-100; i<N; i++)
		err += abs(1.0-L(i,i,N));
	return std::to_string(err);     
} 
