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
__global__ void LU_parallel(int n,float U[M][M], float L[M][M]){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        
    }
}



int main(){
    float U[M][M] = {{1,1,1},{2,3,5},{4,6,8}};
    float L[M][M] = {{1,0,0},{0,1,0},{0,0,1}};
    int n = 0;
    for(int i  = 1, k = M; i<M;i++, k--){
        n += (M-i)*k;
    } 
    


    printf("n calculated: %d\n",n);
    LU_sequential(U, L);

}