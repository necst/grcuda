#include <stdio.h>

#define M 3

void LU_sequential(float U[M][M], float L[M][M]){
    int number = 0;
    for(int k = 0; k<M; k++){
        for(int j = k+1; j<M; j++){
            L[j][k] = U[j][k]/U[k][k];
            for(int v = k; v<M;v++){
                U[j][v] -= L[j][k]*U[k][v];
                number++;
            }
        }
    }
    printf("n sequential: %d\n", number);
    for(int i = 0;i<M;i++){
        for(int j = 0; j<M;j++){
            printf("%f ",L[i][j]);
        }
        printf("\n");
    }
}
// call the inner loop M times
__global__ void LU_parallel(int k,int n, float*U, float*L){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        int v = i/(M-k);
        L[(k+1)*M + v] = U[(k+1)*M + k]/U[k*M + k];
        U[(k+1)*M + v] -= L[(k+1)*M + v]*U[k*M + v];
        printf(" %f\n",L[(k+1)*M + v]);
    }
}



int main(){
    float U[M][M] = {{1,1,1},{2,3,5},{4,6,8}};
    float L[M][M] = {{1,0,0},{0,1,0},{0,0,1}};
    float U_h[M][M] = {{1,1,1},{2,3,5},{4,6,8}};
    float L_h[M][M] = {{1,0,0},{0,1,0},{0,0,1}};
    int n = 0;
    for(int i  = 1, k = M; i<M;i++, k--){
        n += (M-i)*k;
    } 
    float *U_d, *L_d;

    cudaMallocManaged(&U_d,M*M*sizeof(float));
    cudaMallocManaged(&L_d,M*M*sizeof(float));

    for(int i = 0; i<M;i++){
        for(int j = 0; j<M;j++){
            U_d[i*M+j] = U_h[i][j];
            L_d[i*M+j] = L_h[i][j];
        }
    }

    for(int i = 0; i<M*M; i++){
        printf(" %f", U_d[i]);
    }
    int i, j;
    for(int o = 0; o<M*M; o++){
        i = o/M;
        j = o%M;

        if(j == i){
            L_d[o] = float(1);
        }else{
            L_d[o] = float(0);
        }
    }

    for( int k = 0; k < M; k++){
        LU_parallel<<<32,32>>>(k+1, (M-(k+1))*(M-k), U_d, L_d);
        cudaDeviceSynchronize();
    }



    printf("n calculated: %d\n",n);
    // LU_sequential(U, L);
    // for(int i = 0; i<M;i++){
    //     for(int j = 0; j<M;j++){
    //         U_d[i*M + j] = U_h[i][j];
    //         L_d[i*M+j] = L_h[i][j];

    //         printf("%d , %d ", L[i][j], L_d[i*M+j]);
    //     }
    // }
    return 0;
}