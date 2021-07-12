#include "b22.cuh"
#include <cstdio>
#include <time.h>

#define NGPU 2

#define A(i,j,N) A[(i)*N+(j)]
#define U(i,j,N) U[(i)*N+(j)]
#define L(i,j,N) L[(i)*N+(j)]
// #define LT(i,j,N) L[(i)*N+(j)]
// #define A_hw(i,j,N) A_hw[(i)*N+(j)]
// #define U_hw(i,j,N) U_hw[(i)*N+(j)]
// #define L_hw(i,j,N) L_hw[(i)*N+(j)]

// Multi-kernel GPU implementation
// L matrix is transposed -> we obtain 2 upper triangular matrix
__global__ void updateU(const float *A, float *U, float *L, const int dim, const int row){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    //int id = row * dim + j;
    if(j<dim && j >= row){
        U(row,j,dim) = A(row,j,dim);
        for(int k = 0; k<row; k++)
            U(row,j,dim) -= L(k,row,dim)*U(k,j,dim);
    }
}

__global__ void updateL(const float *A, float *U, float *L, const int dim, const int col){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int id = row * dim + j;
    if(i<dim && i>=col){
        L(col,i,dim) = A(i,col,dim);
        for(int k = 0; k<col; k++)
            L(col,i,dim) -= L(k,i,dim)*U(k,col,dim);
        L(col,i,dim) /= U(col,col,dim);
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

// Multi-kernel SW implementation
void SWupdateU(float A[DIM][DIM], float U[DIM][DIM], float L[DIM][DIM], int dim, int row){
    for(int j = row; j<dim; j++){
        U[row][j] = A[row][j];
        for(int k = 0; k<row; k++)
            U[row][j] -= L[row][k]*U[k][j];
    }
}

void SWupdateL(float A[DIM][DIM], float U[DIM][DIM], float L[DIM][DIM], int dim, int col){
    for(int j = col; j<dim; j++){
        L[j][col] = A[j][col];
        for(int k = 0; k<col; k++)
            L[j][col] -= L[j][k]*U[k][col];
        L[j][col] /= U[col][col];
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
        err = cudaStreamCreate(&s[i]);
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
    for (int it = 0; it < N; it++){
        for(int g = 0; g < NGPU; g++){
            cudaSetDevice(g);
            (g%2==0) ? updateU<<<num_blocks,block_size_1d>>>(A, U, L, N, it) : updateL<<<num_blocks,block_size_1d>>>(A, U, L, N, it);
            cudaDeviceSynchronize();
        }
    }
}

void Benchmark22::execute_async(int iter){
    for (int it = 0; it < N; it++){
        cudaSetDevice(0);
        updateU<<<num_blocks,block_size_1d>>>(A, U, L, N, it);
        cudaDeviceSynchronize();
        // err = cudaStreamSynchronize(s[0]);
        // err = cudaStreamSynchronize(s[1]);
        cudaSetDevice(0);
        updateL<<<num_blocks,block_size_1d>>>(A, U, L, N, it);
        cudaDeviceSynchronize();
    }
}
void Benchmark22::execute_cudagraph(int iter){}
void Benchmark22::execute_cudagraph_manual(int iter){}
void Benchmark22::execute_cudagraph_single(int iter){}

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
    std::string res = std::string("\n");
    for(int i = 0; i<N; i=i+N/10)
        res += std::to_string(L(i,i,N)) + "\n";
    return res;    
} 
