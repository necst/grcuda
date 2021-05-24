#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include<sys/time.h>
#include <LU_decomposition.cuh>

//#define N 6

__global__ void add( float *a, float *b, float *c) {
	int tid = blockIdx.x;	//Handle the data at the index

		c[tid] = a[tid] + b[tid];
}


__global__ void scale(float *a, int size, int index){
 	int i;
	int start=(index*size+index);
	int end=(index*size+size);
	
	for(i=start+1;i<end;i++){
		a[i]=(a[i]/a[start]);
	}

}

__global__ void reduce(float *a, int size, int index){
	int i;
       // int tid=threadIdx.x;
	int tid=blockIdx.x;
	int start= ((index+tid+1)*size+index);
	int end= ((index+tid+1)*size+size);

        for(i=start+1;i<end;i++){
                 // a[i]=a[i]-(a[start]*a[(index*size)+i]);
		 	a[i]=a[i]-(a[start]*a[(index*size)+(index+(i-start))]);
        }

}

void LU_decomposition::alloc(){

}

void LU_decomposition::init(){

}
void LU_decomposition::reset(){

}
void LU_decomposition::execute_sync(int iter){

}
void LU_decomposition::execute_async(int iter){


}
void LU_decomposition::execute_cudagraph(int iter){

}
void LU_decomposition::execute_cudagraph_manual(int iter){

}
void LU_decomposition::execute_cudagraph_single(int iter){

}
void LU_decomposition::prefetch(cudaStream_t &s1, cudaStream_t &s2){
    
}
std::string print_result(bool short_form = false);



int main(int argc, char *argv[]){

	float *a;
	float *c;
	int N;
	int flag;

 	float **result;
	float **b;
    float *dev_a, *dev_b, *dev_c;
	int i;
	int j;
	int k;
	float l1;
	float u1;

	double start;
	double end;
	struct timeval tv;
	N=atoi(argv[1]);	
	//allocate memory on CPU
	a=(float *)malloc(sizeof(float)*N*N);
	c=(float *)malloc(sizeof(float)*N*N);


	result=(float **)malloc(sizeof(float *)*N);
	b=(float **)malloc(sizeof(float *)*N);


	for(i=0;i<N;i++){
	   result[i]=(float *)malloc(sizeof(float)*N);
   	   b[i]=(float *)malloc(sizeof(float)*N);
	}

	//allocate the memory on the GPU
	cudaMalloc ( (void**)&dev_a, N*N* sizeof (float) );
	cudaMalloc ( (void**)&dev_b, N*N* sizeof (float) );
	cudaMalloc ( (void**)&dev_c, N*N* sizeof (float) );


	srand((unsigned)2);
	//fill the arrays 'a' and 'b' on the CPU
	for ( i = 0; i <= (N*N); i++) {
		a[i] =((rand()%10)+1);
	}
	


	cudaMemcpy( dev_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice);//copy array to device memory
       
 	gettimeofday(&tv,NULL);
	start=tv.tv_sec;
        /*Perform LU Decomposition*/
      	printf("\n=========================================================="); 
	for(i=0;i<N;i++){
        scale<<<1,1>>>(dev_a,N,i);
		reduce<<<(N-i-1),1>>>(dev_a,N,i);


    }
        /*LU decomposition ends here*/


	cudaMemcpy( c, dev_a, N*N*sizeof(float),cudaMemcpyDeviceToHost );//copy array back to host

	printf("\nThe time for LU decomposition is %lf \n",(end-start));


	/*copy the result matrix into explicit 2D matrix for verification*/
	for(i=0;i<N;i++){
			for(j=0;j<N;j++){
			result[i][j]=c[i*N+j];
		}
	}


	printf("=======================================================");
	printf("\n Performing inplace verification \n");
        /*Inplace verification step*/

	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			b[i][j]=0;
			for(k=0;k<N;k++){
				if(i>=k)l1=result[i][k];
				else l1=0;

				if(k==j)u1=1;
				else if(k<j)u1=result[k][j];//figured it out 
				else u1=0.0;

			b[i][j]=b[i][j]+(l1*u1);

			}
		}
		}


	printf("\n==================================================\n");
	// printf("\nThe b matrix\n");	
	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			if(abs(a[i*N+j]-b[i][j])>0.5) flag=flag+1;	
		}
	}

	if(flag==0) printf("Match\n");
	else printf("No match \n");



	//free the memory allocated on the GPU
	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );
	
	return 0;
}



