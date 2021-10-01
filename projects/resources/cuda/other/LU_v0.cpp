#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <chrono>
#include <ctime>
#include <unistd.h>
#include <stdlib.h>

#define M 3
#define x(i,j) x[(i)*M+(j)]

// hp: matrici U e L inizializzate a 0 
void LU_sequential(float A[M][M], float U[M][M], float L[M][M], int dim){
    for(int i = 0; i<dim; i++){
        for(int j = i; j<dim; j++){
            U[i][j] = A[i][j];
            L[j][i] = A[j][i];
            for(int k = 0; k<i; k++){
                U[i][j] -= L[i][k]*U[k][j];
                L[j][i] -= L[j][k]*U[k][i];
            }
            L[j][i] /= U[i][i];
        }
    }
}

void updateU(float A[M][M], float U[M][M], float L[M][M], int dim, int row){
    for(int j = row; j<dim; j++){
        U[row][j] = A[row][j];
        for(int k = 0; k<row; k++)
            U[row][j] -= L[row][k]*U[k][j];
    }
}

void updateL(float A[M][M], float U[M][M], float L[M][M], int dim, int col){
    for(int j = col; j<dim; j++){
        L[j][col] = A[j][col];
        for(int k = 0; k<col; k++)
            L[j][col] -= L[j][k]*U[k][col];
        L[j][col] /= U[col][col];
    }
}

void initializeA(float A[M][M], float A_h[M][M], int dim, int seed){
    srand(seed);
    for(int i = 0; i<dim; i++)
        for(int j = 0; j<dim; j++){
            A[i][j] = rand()%100 + 1;
            if(rand()%2)
                A[i][j] *= -1;
            A_h[i][j] = A[i][j];
        }
}

// call the inner loop M times
/*
__global__ void LU_parallel(int k,int n, float*U, float*L){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        int v = i%M + k;
        int j = i/M + 1 + k;
        if((v-k) == 0){
            L[j*M + k] = U[j*M + k]/U[k*M + k];
        }
        U[j*M + v] -= L[j*M + k]*U[k*M + v];
    }
}

*/

int main(){

    int dim = 3;
    //float A[M][M] = {{3,-7,-2,2},{-3,5,1,0},{6,-4,0,-5},{-9,5,-5,12}};
    float A[M][M] = {{0}};
    float U[M][M] = {{0}};
    float L[M][M] = {{0}};
    float A_h[M][M] = {{0}};
    float U_h[M][M] = {{0}};
    float L_h[M][M] = {{0}};
    std::chrono::time_point<std::chrono::steady_clock> StartTime;
    std::chrono::time_point<std::chrono::steady_clock> EndTime;
    initializeA(A, A_h, dim, 13);

    // float *U_d, *L_d;

    // cudaMallocManaged(&U_d,M*M*sizeof(float));
    // cudaMallocManaged(&L_d,M*M*sizeof(float));

    // for(int i = 0; i<M;i++){
    //     for(int j = 0; j<M;j++){
    //         U_d[i*M+j] = U_h[i][j];
    //         L_d[i*M+j] = L_h[i][j];
    //     }
    // }

    // for( int k = 0; k < M; k++){
    //     LU_parallel<<<32,32>>>(k, (M-(k+1))*(M-k), U_d, L_d);
    //     cudaDeviceSynchronize();
    // }

    printf("\nSingle Kernel\n");
    StartTime = std::chrono::steady_clock::now();
    LU_sequential(A, U, L, dim);
    EndTime = std::chrono::steady_clock::now();
    printf("Elapsed Time = %lli", std::chrono::duration_cast<std::chrono::microseconds>(EndTime - StartTime).count());

    if(dim<=10){
        printf("\nL = \n");
        for(int i = 0; i<dim;i++){
            for(int j = 0; j<dim;j++)
                printf("%.2f ", L[i][j]);
            printf("\n");
        }
        
        printf("\nU = \n");
        for(int i = 0; i<dim;i++){
            for(int j = 0; j<dim;j++){
                printf("%.2f ", U[i][j]);
            }
            printf("\n");
        }
    }

    printf("\nMulti-kernel\n");
    StartTime = std::chrono::steady_clock::now();
    for(int i = 0; i<dim;i++){
        updateU(A_h, U_h, L_h, dim, i);
        updateL(A_h, U_h, L_h, dim, i);
    }
    EndTime = std::chrono::steady_clock::now();
    printf("Elapsed Time = %lli", std::chrono::duration_cast<std::chrono::microseconds>(EndTime - StartTime).count());
    if(dim<=10){
        printf("\nL_h = \n");
        for(int i = 0; i<dim;i++){
            for(int j = 0; j<dim;j++)
                printf("%.2f ", L_h[i][j]);
            printf("\n");
        }
        
        printf("\nU_h = \n");
        for(int i = 0; i<dim;i++){
            for(int j = 0; j<dim;j++)
                printf("%.2f ", U_h[i][j]);
            printf("\n");
        }
    }

    int correct = 1;
    for(int i = 0; i<dim && correct; i++){
        for(int j = 0; j<dim && correct; j++)
            if(U[i][j]!=U_h[i][j] || L[i][j]!=L_h[i][j])
                correct = 0;
    }
    
    if(!correct) 
        printf("\nERROR: single and multi-kernel results are not equal :(\n");
    else 
        printf("\nCORRECT: single and multi-kernel results are equal :)\n"); 
    
    return 0;
}