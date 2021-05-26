//https://andrewbolster.info/2011/04/lu-decomposition-in-c-and-under-cuda.html


#include <utility>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
 
#define CUDA_CHK(NAME, ARGS) { \
  cudaError_t cuda_err_code = NAME ARGS; \
  if (cuda_err_code != cudaSuccess) { \
    printf("%s failed with code %d\n", #NAME, cuda_err_code); \
    abort(); \
  } \
}
#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif
 
#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif
#define MAT1 3
#define MAT2 MAT1*MAT1
#define TINY 1.0e-40
#define a(i,j) a[(i)*MAT1+(j)]
 
#define GO 1
#define NOGO 0
 
void Check_Kernel(const char *message){
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess){
    fprintf(stderr,"Error: %s:%s\n",message, cudaGetErrorString(error));
  }
}
 
__device__ void d_pivot_decomp(float *a, int *p, int *q){
    int i,j,k;
    int n=MAT1;
    int pi,pj,tmp;
    float max;
    float ftmp;
    for (k=0;k<n;k++){
        pi=-1,pj=-1,max=0.0;
        //find pivot in submatrix a(k:n,k:n)
        for (i=k;i<n;i++) {
            for (j=k;j<n;j++) {
                if (fabs(a(i,j))>max){
                    max = fabs(a(i,j));
                    pi=i;
                    pj=j;
                }
            }
        }
        //Swap Row
        tmp=p[k];
        p[k]=p[pi];
        p[pi]=tmp;
        for (j=0;j<n;j++){
            ftmp=a(k,j);
            a(k,j)=a(pi,j);
            a(pi,j)=ftmp;
        }
        //Swap Col
        tmp=q[k];
        q[k]=q[pj];
        q[pj]=tmp;
        for (i=0;i<n;i++){
            ftmp=a(i,k);
            a(i,k)=a(i,pj);
            a(i,pj)=ftmp;
        }
        //END PIVOT
 
        //check pivot size and decompose
        if ((fabs(a(k,k))>TINY)){
            for (i=k+1;i<n;i++){
                //Column normalisation
                ftmp=a(i,k)/=a(k,k);
                for (j=k+1;j<n;j++){
                    //a(ik)*a(kj) subtracted from lower right submatrix elements
                    a(i,j)-=(ftmp*a(k,j));
                }
            }
        }
        //END DECOMPOSE
    }
}
 
 
__device__ void d_solve(float *a, float *x, int *p, int *q){
    //forward substitution; see  Golub, Van Loan 96
    //And see http://www.cs.rutgers.edu/~richter/cs510/completePivoting.pdf
    int i,ii=0,j;
    float ftmp;
    float xtmp[MAT1];
    //Swap rows (x=Px)
    for (i=0; i<MAT1; i++){
        xtmp[i]=x[p[i]]; //value that should be here
    }
    //Lx=x
    for (i=0;i<MAT1;i++){
        ftmp=xtmp[i];
        if (ii != 0)
            for (j=ii-1;j<i;j++)
                ftmp-=a(i,j)*xtmp[j];
        else
            if (ftmp!=0.0)
                ii=i+1;
        xtmp[i]=ftmp;
    }
    //backward substitution
    //partially taken from Sourcebook on Parallel Computing p577
    //solves Uy=z
    xtmp[MAT1-1]/=a(MAT1-1,MAT1-1);
    for (i=MAT1-2;i>=0;i--){
        ftmp=xtmp[i];
        for (j=i+1;j<MAT1;j++){
            ftmp-=a(i,j)*xtmp[j];
        }
        xtmp[i]=(ftmp)/a(i,i);
    }
    for (i=0;i<MAT1;i++)
 
    //Last bit
    //solves x=Qy
    for (i=0;i<MAT1;i++){
        x[i]=xtmp[q[i]];
    }
}
 
__global__ void solve(float *A, float *B, int max){
  //Each thread solves the A[id]x[id]=b[id] problem
  int id= blockDim.x*blockIdx.x + threadIdx.x;
  int p_pivot[MAT1],q_pivot[MAT1];
  if ((GO==1) && (id < max)){
    for (int i=0;i<MAT1;i++) {
        p_pivot[i]=q_pivot[i]=i;
    }
 
    d_pivot_decomp(&A[id*MAT2],&p_pivot[0],&q_pivot[0]);
    d_solve(&A[id*MAT2],&B[id*MAT1],&p_pivot[0],&q_pivot[0]);
  }
}
 
/*
void notmain(){
    //3x3 Matrix
    //float a[]={1,-2,3,2,-5,12,0,2,-10};
    float a[]={1,3,-2,3,5,6,2,4,3};
    float b[]={5,7,8};
    //float a[]={1,2,3,2,-1,1,3,4,-1};
    //float b[]={14,3,8};
    //float a[]={1,-2,1,0,2,2,-2,4,2};
    //float b[]={1,4,2};
    int sig;
    puts("Declared Stuff");
 
    //pivot array (not used currently)
    int* p_pivot = (int *)malloc(sizeof(int)*MAT1);
    int* q_pivot = (int *)malloc(sizeof(int)*MAT1);
    puts("Starting Stuff");
    for (unsigned int i=0; i<MAT1; i++){
        p_pivot[i]=i;
        q_pivot[i]=i;
        printf("%.1lf|",b[i]);
        for (unsigned int j=0;j<MAT1; j++){
            printf("%.1lf,",a(i,j));
        }
        printf("|%d,%d",p_pivot[i],q_pivot[i]);
        puts("");
    }
 
    h_pivot_decomp(&a[0],p_pivot,q_pivot);
    puts("After Pivot");
    for (unsigned int i=0; i<MAT1; i++){
        printf("%.1lf|",b[i]);
        for (unsigned int j=0;j<MAT1; j++){
            printf("%.1lf,",a(i,j));
        }
        printf("|%d,%d",p_pivot[i],q_pivot[i]);
        puts("");
    }
 
    h_solve(&a[0],&b[0],p_pivot,q_pivot);
    puts("Finished Solve");
 
    for (unsigned int i=0; i<MAT1; i++){
        printf("%.1lf|",b[i]);
        for (unsigned int j=0;j<MAT1; j++){
            printf("%.1lf,",a(i,j));
        }
        puts("");
    }
}*/
 
 
 
int main(){
  //What are you actually trying to do:
  //  generate 2 input matrixes, (NxN,Nx1) and 1 output (1xN)
  //  do this over matrixcount length for threadiding
  const unsigned int matrixcount=1;
  const unsigned int matsize=MAT2*matrixcount;
  const unsigned int vecsize=MAT1*matrixcount;
  float a[]={1,3,-2,3,5,6,2,4,3};
  //const float exampleA[]={7,3,-11,-6,7,10,-11,2,-2};
  //const float exampleA[]={4,3,6,3};
  const float b[]={5,7,8};
  //const float exampleB[]={4,5};
 
  //memory allocations
  float* h_A = (float*)malloc(sizeof(float)*matsize);
  float* h_b = (float*)malloc(sizeof(float)*vecsize);
  float* h_x = (float*)malloc(sizeof(float)*vecsize);
 
  float* d_A;
  float* d_b;
  float* d_x;
  CUDA_CHK(cudaMalloc, (&d_A, sizeof(float)*matsize));
  CUDA_CHK(cudaMalloc, (&d_b, sizeof(float)*vecsize));
  CUDA_CHK(cudaMalloc, (&d_x, sizeof(float)*vecsize));
 
  printf("Mallocd\n");
  //fill matrix and vector with stuff
  for (unsigned int i = 0;i<matrixcount;i++){
    //printf("\n%d\n",i);
    for (unsigned int j = 0; j < MAT1; j++){
      h_b[(i*MAT1)+j]=b[j];
      h_x[(i*MAT1)+j]=-1;
      printf("%.0f,",h_b[(i*MAT1)+j]);
      //printf("\n%d:",j);
      for (unsigned int k=0; k < MAT1; k++){
        //printf("%d,",k);
        h_A[(i*MAT2)+(j*MAT1)+k]=a(j,k);
      }
    }
    puts("\n");
  }
 
  printf("Generated\n");
    //copy values to device
  CUDA_CHK(cudaMemcpy, (d_A, h_A, sizeof(float)*matsize, cudaMemcpyHostToDevice));
  CUDA_CHK(cudaMemcpy, (d_b, h_b, sizeof(float)*vecsize, cudaMemcpyHostToDevice));
  CUDA_CHK(cudaMemcpy, (d_x, h_x, sizeof(float)*vecsize, cudaMemcpyHostToDevice));
 
  printf("Copied\n");
  for (unsigned int i=0; i<matrixcount; i++){
    printf("\n%d:x:A|B",i);
    //printf("%.3lf|",h_x[i*MAT1]);
    for (unsigned int j=0; j<MAT1; j++){
      printf("\n%.3lf:",h_x[i*MAT1+j]);
      for (unsigned int k=0;k<MAT1; k++){
        printf("%.1lf,",h_A[(i*MAT2)+(j*MAT1)+k]);
      }
      printf("|%.3lf",h_b[i*MAT1+j]);
    }
  }
  puts("\n");
 
  //parameters
  dim3 threadsPerBlock(1,1,1);
  dim3 blocksPerGrid((matrixcount + threadsPerBlock.x -1)/threadsPerBlock.x,1,1);
  printf("TPB:%d,BPG:%d\n",threadsPerBlock.x,blocksPerGrid.x);
  //Execute
  cudaEvent_t evt_start, evt_stop;
  CUDA_CHK(cudaEventCreate, (&evt_start));
  CUDA_CHK(cudaEventCreate, (&evt_stop));
  CUDA_CHK(cudaEventRecord, (evt_start,0));
 
  solve<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_b,matrixcount);
  cudaDeviceSynchronize();
  Check_Kernel("Solve");
 
  printf("Ran solve\n");
  CUDA_CHK(cudaEventRecord, (evt_stop, 0));
  CUDA_CHK(cudaEventSynchronize, (evt_stop));
  float total_time;
  CUDA_CHK(cudaEventElapsedTime, (&total_time, evt_start, evt_stop));
  CUDA_CHK(cudaMemcpy, (h_A,d_A, sizeof(float)*matsize, cudaMemcpyDeviceToHost));
  CUDA_CHK(cudaMemcpy, (h_x,d_b, sizeof(float)*vecsize, cudaMemcpyDeviceToHost));
 
  // print timing results
  float one_time = total_time * 1e-3;
 
  printf("time: %g s\n", one_time);
  for (unsigned int i=0; i<matrixcount; i++){
    printf("\n%d:x:A",i);
    //printf("%.3lf|",h_x[i*MAT1]);
    for (unsigned int j=0; j<MAT1; j++){
      printf("\n%.3lf:",h_x[i*MAT1+j]);
      for (unsigned int k=0;k<MAT1; k++){
        printf("%.1lf,",h_A[(i*MAT2)+(j*MAT1)+k]);
      }
    }
  }
  puts("\n");
 
  cudaEventDestroy(evt_start);
  cudaEventDestroy(evt_stop);
  free(h_A);
  free(h_b);
  free(h_x);
  CUDA_CHK(cudaFree, (d_A)); 
  CUDA_CHK(cudaFree, (d_x)); 
  //CUDA_CHK(cudaFree, (d_pivot)); 
}