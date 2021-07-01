#include "b22.cuh"
#include <cstdio>

#define IT 1001
#define NGPU 1
#define DIM 100
#define A(i,j) A[(i)*DIM+(j)]
#define U(i,j) U[(i)*DIM+(j)]
#define L(i,j) L[(i)*DIM+(j)]
#define A_hw(i,j) A_hw[(i)*DIM+(j)]
#define U_hw(i,j) U_hw[(i)*DIM+(j)]
#define L_hw(i,j) L_hw[(i)*DIM+(j)]

__global__ void LU_decomposition(const float *A, float *U, float *L, const int dim){
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    //int id = i*dim + j;

    if(i<dim && j <dim){
        U(i,j) = A(i,j);
        L(j,i) = A(j,i);
        for(int k = 0; k<i; k++){
            U(i,j) -= L(i,k)*U(k,j);
            L(j,i) -= L(j,k)*U(k,i);
        }
        L(j,i) /= U(i,i);
    }
}

__global__ void updateU(const float *A, float *U, const float *L, const int dim, const int row){
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    //int id = row * dim + j;
    if(j<dim){
        U(row,j) = A(row,j);
        for(int k = 0; k<row; k++)
            U(row,j) -= L(row,k)*U(k,j);
    }
}


__global__ void updateL(const float *A, const float *U, float *L, const int dim, const int col){
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    //int id = row * dim + j;
    if(i<dim){
        L(i,col) = A(i,col);
        for(int k = 0; k<col; k++)
            L(i,col) -= L(i,k)*U(k,col);
        L[i][col] /= U(col,col);
    }
}

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

void initializeA(float A[DIM][DIM], int dim, int seed){
    srand(seed);
    for(int i = 0; i<dim; i++)
        for(int j = 0; j<dim; j++){
            A[i][j] = rand()%100 + 1;
            if(rand()%2)
                A[i][j] *= -1;
        }
}
  
void Benchmark22::alloc(){
    cudaMallocManaged(&A_hw, long(N)*long(N)*sizeof(float));
    cudaMallocManaged(&U_hw, long(N)*long(N)*sizeof(float));
    cudaMallocManaged(&L_hw, long(N)*long(N)*sizeof(float));
}

void Benchmark22::init(){
    for(int i = 0; i<N; i++){
        for(int j = 0; j<N; j++){
            U[i][j] = 0.0;
            L[i][j] = 0.0;
            A[i][j] = float(rand()%100 + 1);
            if(rand()%2)
                A[i][j] *= -1;
            A_hw(i,j) = A[i][j];
            U_hw(i,j) = 0.0;
            L_hw(i,j) = 0.0;
        }
    }
}

void Benchmark22::reset(){
    for(int i = 0; i<N; i++){
        for(int j = 0; j<N; j++){
            U[i][j] = 0.0;
            L[i][j] = 0.0;
            U_hw(i,j) = 0.0;
            L_hw(i,j) = 0.0;
        }
    }
}

void Benchmark22::execute_sync(int iter){

    printf("Executing SW version");
    for (int it = 0; it < N; it++){
        SWupdateU(A, U, L, N, it);
        SWupdateL(A, U, L, N, it);
    }

    printf("Executing HW version");
    cudaSetDevice(0);
    for (int it = 0; it < N; it++){
        cudaDeviceSynchronize();
        updateU<<<1024,32>>>(A_hw, U_hw, L_hw, N, it);
        //JacobiIterationDistributed_v2<<<1024, 32>>>(N, a_d, (it%2==0)?x_d:x_result_d, 0, N/NGPU, b_d, (it%2==0)?x_result_d:x_d);

        //printf("switch device\n");
        
        //cudaSetDevice(1);
        cudaDeviceSynchronize();
        updateL<<<1024,32>>>(A_hw, U_hw, L_hw, N, it);

        //JacobiIterationDistributed_v2<<<1024, 32>>>(N, a_d, (it%2==0)?x_d:x_result_d, N/NGPU, N, b_d, (it%2==0)?x_result_d:x_d);
        
    }
}

void Benchmark22::execute_async(int iter){
    /*
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
    */
}
void Benchmark22::execute_cudagraph(int iter){}
void Benchmark22::execute_cudagraph_manual(int iter){}
void Benchmark22::execute_cudagraph_single(int iter){}

std::string Benchmark22::print_result(bool short_form){
    std::string res;
    int correct = 1;
    for(int i = 0; i<N && correct; i++){
        for(int j = 0; j<N && correct; j++)
            if(U[i][j]!=U_hw(i,j) || L[i][j]!=L_hw(i,j))
                correct = 0;
    }
    if(!correct) 
        res = "\nERROR: sw and hw results are not equal :(\n";
    else 
        res = "\nCORRECT: sw and hw results are equal :)\n"; 
    
    return res;
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