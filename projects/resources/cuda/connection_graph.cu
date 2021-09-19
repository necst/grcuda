#include <iostream>
#include <stdio.h>
#include <nvml.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <fstream>

#define N 500000000 // 500 MB

float D2D_copy (size_t size, int from, int to){
	int *pointers[2];

	cudaSetDevice (from);
	cudaDeviceEnablePeerAccess (to, 0);
	cudaMalloc (&pointers[0], size);

	cudaSetDevice (to);
	cudaDeviceEnablePeerAccess (from, 0);
	cudaMalloc (&pointers[1], size);

	cudaEvent_t begin, end;
	cudaEventCreate (&begin);
	cudaEventCreate (&end);

	cudaEventRecord (begin);
	cudaMemcpyAsync (pointers[0], pointers[1], size, cudaMemcpyDeviceToDevice);
	cudaEventRecord (end);
	cudaEventSynchronize (end);

	float elapsed;
	cudaEventElapsedTime (&elapsed, begin, end);
	elapsed /= 1000;

	cudaSetDevice (from);
	cudaFree (pointers[0]);

	cudaSetDevice (to);
	cudaFree (pointers[1]);

	cudaEventDestroy (end);
	cudaEventDestroy (begin);
	cudaSetDevice (from);

	return elapsed;
}

float HToD_copy (size_t size, int deviceID){
	int *pointer, *d_pointer;

	cudaSetDevice (deviceID);
	cudaMalloc (&d_pointer, size);
	cudaMallocHost(&pointer, size);

	cudaEvent_t begin, end;
	cudaEventCreate (&begin);
	cudaEventCreate (&end);

	cudaEventRecord (begin);
	cudaMemcpyAsync (d_pointer, pointer, size, cudaMemcpyHostToDevice);
	cudaEventRecord (end);
	cudaEventSynchronize (end);

	float elapsed;
	cudaEventElapsedTime (&elapsed, begin, end);
	elapsed /= 1000;

	cudaSetDevice (deviceID);
	cudaFree (d_pointer);

	cudaEventDestroy (end);
	cudaEventDestroy (begin);

	return elapsed;
}

int main(){
	size_t size = N; 
	unsigned int NGPU {};

	nvmlInit ();
	nvmlDeviceGetCount (&NGPU);  
	printf("N_Devices = %d\n", NGPU);

	double **bands = (double**)malloc(NGPU * sizeof(double*));
	for(int i=0; i<NGPU; i++)
		bands[i] = (double*)malloc(NGPU * sizeof(double));

	std::ofstream out_file;
	
	out_file.open("/usr/tmp/GrCUDA/connection_graph.csv");
	out_file << "From,To,Bandwidth\n";

	for(int i=0; i<NGPU; i++){
		double time_H2D = HToD_copy(size, 1);
		printf("\nfrom: Host, to: %d, time spent: %f, transfer rate: %f GB/s \n",i, time_H2D, (float(N)/1000000000.0)/time_H2D);
		out_file<< std::setprecision (15) << "-1" << "," << i << "," <<(double(N)/1000000000.0)/time_H2D << "\n";
		for(int j=0 ; j<NGPU; j++){
			double time_D2D = D2D_copy(size, i, j);
			bands[i][j] = (double(N)/1000000000.0)/time_D2D;
			printf("from: %d, to: %d, time spent: %f, transfer rate: %f GB/s \n",i, j, time_D2D, bands[i][j]);
			out_file << i << "," << j << "," << bands[i][j] << "\n";
		}
	}
	
	out_file.close();

	return 0;
}
