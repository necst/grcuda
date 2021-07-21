#include "b22.cuh"
#include <cstdio>
#include <time.h>

//#define NGPU 4
#define GPU0 0
#define GPU1 1
#define N_STREAMS 1

#define A(i,j,N) A[(i)*N+(j)]
#define U(i,j,N) U[(i)*N+(j)]
#define L(i,j,N) L[(i)*N+(j)]
// #define LT(i,j,N) L[(i)*N+(j)]
// #define A_hw(i,j,N) A_hw[(i)*N+(j)]
// #define U_hw(i,j,N) U_hw[(i)*N+(j)]
// #define L_hw(i,j,N) L_hw[(i)*N+(j)]

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

/*
// Multi-kernel SW implementation
// L matrix is transposed -> we obtain 2 upper triangular matrix
void SWupdateUT(float A[DIM][DIM], float U[DIM][DIM], float LT[DIM][DIM], int dim, int row){
	for(int j = row; j<dim; j++){
		U[row][j] = A[row][j];
		for(int k = 0; k<row; k++)
			U[row][j] -= LT[k][row]*U[k][j];
	}
}

void SWupdateLT(float A[DIM][DIM], float U[DIM][DIM], float LT[DIM][DIM], int dim, int row){
	for(int j = row; j<dim; j++){
		LT[row][j] = A[j][row];
		for(int k = 0; k<row; k++)
			LT[row][j] -= LT[k][j]*U[k][row];
		LT[row][j] /= U[row][row];
	}
}

// initialize A with random numbers in [-99, 99]
void initializeA(float A[DIM][DIM], int dim, int seed){
	srand(seed);
	for(int i = 0; i<dim; i++)
		for(int j = 0; j<dim; j++){
			A[i][j] = rand()%100 + 1;
			if(rand()%2)
				A[i][j] *= -1;
		}
}
*/
  
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

/*
void Benchmark22::init(){
	for(int i = 0; i<N; i++){
		for(int j = 0; j<N; j++){
			U[i][j] = 0.0;
			LT[i][j] = 0.0;
			A[i][j] = float(rand()%100 + 1);
			if(rand()%2)
				A[i][j] *= -1;
			A_hw(i,j,N) = A[i][j];
			U_hw(i,j,N) = 0.0;
			L_hw(i,j,N) = 0.0;
		}
	}
}

void Benchmark22::reset(){
	srand(time(0));
	for(int i = 0; i<N; i++){
		for(int j = 0; j<N; j++){
			U[i][j] = 0.0;
			LT[i][j] = 0.0;
			A[i][j] = float(rand()%100 + 1);
			if(rand()%2)
				A[i][j] *= -1;
			A_hw(i,j,N) = A[i][j];
			U_hw(i,j,N) = 0.0;
			L_hw(i,j,N) = 0.0;
		}
	}
}

void Benchmark22::execute_sync(int iter){
	
	printf("\nExecuting SW version (transpose)");
	for (int it = 0; it < N; it++){
		SWupdateUT(A, U, LT, N, it);
		SWupdateLT(A, U, LT, N, it);
	}
	
	printf("\nExecuting on GPU\n");
	cudaSetDevice(0);
	for (int it = 0; it < N; it++){
		updateU<<<num_blocks,block_size_1d>>>(A, U, L, N, it);
		cudaDeviceSynchronize();
		updateL<<<num_blocks,block_size_1d>>>(A, U, L, N, it);
		cudaDeviceSynchronize();
	}
}
*/

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
	for (int it = 0; it < N; it++){
		for(int g = 0; g < NGPU; g++){
			cudaSetDevice(g);
			for (int i = 0; i < N_STREAMS; i++){
				int offset = g*slice_GPU+i*slice_stream;
				int max = (g*slice_GPU+(i+1)*slice_stream < (g+1)*slice_GPU) ? g*slice_GPU+(i+1)*slice_stream : (g+1)*slice_GPU;
				// updateU<<<(N/block_size_1d+1)/NGPU+1,block_size_1d,0,s[g*N_STREAMS+i]>>>(A, U, L, offset, max, N, it);
				updateU<<<num_blocks,block_size_1d,0,s[g*N_STREAMS+i]>>>(A, U, L, offset, (max<N) ? max:N, N, it);
				err = cudaStreamSynchronize(s[g*N_STREAMS+i]);
			}
			// cudaDeviceSynchronize();
		}
		for(int g = 0; g < NGPU; g++){
			cudaSetDevice(g);
			for (int i = 0; i < N_STREAMS; i++){
				int offset = g*slice_GPU+i*slice_stream;
				int max = (g*slice_GPU+(i+1)*slice_stream < (g+1)*slice_GPU) ? g*slice_GPU+(i+1)*slice_stream : (g+1)*slice_GPU;
				// updateL<<<(N/block_size_1d+1)/NGPU+1,block_size_1d,0,s[g*N_STREAMS+i]>>>(A, U, L, offset, max, N, it); 
				updateL<<<num_blocks,block_size_1d,0,s[g*N_STREAMS+i]>>>(A, U, L, offset, (max<N) ? max:N, N, it);
				err = cudaStreamSynchronize(s[g*N_STREAMS+i]);
			}
			// cudaDeviceSynchronize();
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

	// if (pascalGpu && do_prefetch) {
	// 	for (int i = 0; i < NGPU; i++) {
	// 		cudaMemPrefetchAsync(A, long(N)*long(N)*sizeof(float), i, s[i]);
	// 		cudaMemPrefetchAsync(L, long(N)*long(N)*sizeof(float), i, s[i]);
	// 		cudaMemPrefetchAsync(U, long(N)*long(N)*sizeof(float), i, s[i]);
	// 	}
    // }

	cudaEvent_t e1, e2;
	int slice_GPU = N/NGPU+1;
	int slice_stream = slice_GPU/N_STREAMS+1;

	for (int it = 0; it < N; it++){

		for(int g = 0; g < NGPU; g++){		
			cudaSetDevice(g);

			for (int i = 0; i < N_STREAMS; i++){
				int offset = g*slice_GPU+i*slice_stream;
				int max = (g*slice_GPU+(i+1)*slice_stream < (g+1)*slice_GPU) ? g*slice_GPU+(i+1)*slice_stream : (g+1)*slice_GPU;
				updateU<<<num_blocks,block_size_1d,0,s[g*N_STREAMS+i]>>>(A, U, L, offset, (max<N) ? max:N, N, it);
				// if(it/slice_stream == g*N_STREAMS+i){ 
				// 	cudaEventCreate(&e1);
				// 	cudaEventRecord(e1, s[g*N_STREAMS+i]);
				// }
			}
			cudaDeviceSynchronize();
		}

		// for(int g = 0; g < NGPU*N_STREAMS; g++){
		// 	// cudaStreamWaitEvent(s[g], e1, 0);
		// 	err = cudaStreamSynchronize(s[g]);
		// }

        for(int g = 0; g < NGPU; g++){		
			cudaSetDevice(g);
			for (int i = 0; i < N_STREAMS; i++){
				int offset = g*slice_GPU+i*slice_stream;
				int max = (g*slice_GPU+(i+1)*slice_stream < (g+1)*slice_GPU) ? g*slice_GPU+(i+1)*slice_stream : (g+1)*slice_GPU;
				updateL<<<num_blocks,block_size_1d,0,s[g*N_STREAMS+i]>>>(A, U, L, offset, (max<N) ? max:N, N, it);
				// if(it/slice_stream == g*N_STREAMS+i){ 
				// 	cudaEventCreate(&e2);
				// 	cudaEventRecord(e2, s[g*N_STREAMS+i]);
				// }
			}
			cudaDeviceSynchronize();
		}

		// for(int g = 0; g < NGPU*N_STREAMS; g++){
		// 	// cudaStreamWaitEvent(s[g], e2, 0);
		// 	err = cudaStreamSynchronize(s[g]);
		// }
	}

	// for (int j = 0; j < NGPU; j++) {
	// 	cudaSetDevice(j);
	// 	cudaDeviceSynchronize();
	// 	// err = cudaStreamSynchronize(s[j]);
	// }

}

void Benchmark22::execute_cudagraph(int iter){}
void Benchmark22::execute_cudagraph_manual(int iter){}
void Benchmark22::execute_cudagraph_single(int iter){
	int ngpu = 1;
	for (int it = 0; it < N; it++){
		for(int g = 0; g < ngpu; g++){
			cudaSetDevice(g);
			updateU<<<(N/block_size_1d+1)/ngpu+1,block_size_1d>>>(A, U, L, g*(N/ngpu+1),(g+1)*(N/ngpu+1), N, it);
			cudaDeviceSynchronize();
		}
		for(int g = 0; g < ngpu; g++){
			cudaSetDevice(g);
			updateL<<<(N/block_size_1d+1)/ngpu+1,block_size_1d>>>(A, U, L, g*(N/ngpu+1),(g+1)*(N/ngpu+1), N, it);
			cudaDeviceSynchronize();
		}
	}
}

std::string Benchmark22::print_result(bool short_form){
	/*
	
	float tolerance = 0.0001;
	for(int i = 0; i<N; i++){
		for(int j = 0; j<N; j++){
			if(abs(U[i][j]-U_hw(i,j,N)) >= tolerance){
				tolerance = abs(U[i][j]-U_hw(i,j,N));
				// printf("(%d,%d) \t U= %f vs %f\terror = %f\n", i,j,U[i][j],U_hw(i,j,N),abs(U[i][j]-U_hw(i,j,N)));
			}
			if(abs(LT[i][j]-L_hw(i,j,N)) >= tolerance){
				tolerance = abs(LT[i][j]-L_hw(i,j,N));
				// printf("(%d,%d) \t LT= %f vs %f\terror = %f\n", i,j,LT[i][j],L_hw(i,j,N),abs(LT[i][j]-L_hw(i,j,N)));
			}
		}
	}
	printf("\n\n");
	*/
	std::string res;
	float err = 0.0;
	for(int i = N-100; i<N; i++)
		err += abs(1.0-L(i,i,N));
	return std::to_string(err);     
} 
