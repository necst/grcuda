#include <omp.h>
#include <cmath>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define ITERATION_LIMIT 333
#define EPSILON 0.0000001
#define NUM_OF_THREADS 4
#define N 100
#define IT 3
bool check_diagoanally_dominant_sequential(float** matrix, int matrix_size);
bool check_diagoanally_dominant_parallel(float** matrix, int matrix_size);
void solve_jacobi_sequential(float* matrix, int matrix_size, float* right_hand_side);
void solve_jacobi_parallel(float** matrix, int matrix_size, float* right_hand_side);
void init_array_sequential(float array[], int array_size);
float* clone_array_sequential(float array[], int array_length);
void init_array_parallel(float array[], int array_size);
float* clone_array_parallel(float array[], int array_length);
void delete_matrix(float* matrix, int matrix_size);

void initAMatrix(int n, float*a){
    int i;
    int j;    
    for ( j = 0; j < n; j++ )
    {
        for ( i = 0; i < n; i++ ){
            if ( j == i - 1 ){
                a[i+j*n] = -1.0;
            }
            else if ( j == i ){
                a[i+j*n] = 2.0;
            }
            else if ( j == i + 1 ){
                a[i+j*n] = -1.0;
            }
            else{
                a[i+j*n] = 0.0;
            }
        }
    }
      
}

void swap(float* &a, float* &b){
    float *temp = a;
    a = b;
    b = temp;
}
int main(){

	
	// Initializing the main structures .. 
	float* matrix = new float[N*N];

	float* right_hand_side = new float[N];

	// init matrix
	initAMatrix(N, matrix);



    // init
    for ( int i = 0; i < N; i++ )
    {
        right_hand_side[i] = 3.0;
    }
    right_hand_side[N-1] = ( float ) ( N + 1 );

	

	// Computing the time
	const clock_t serial_starting_time = clock();
	solve_jacobi_sequential(matrix, N, right_hand_side);
	printf("Elapsed time: %f ms\n", float(clock() - serial_starting_time));


	// // Computing the time
	// const clock_t parallel_starting_time = clock();
	// // Initializing the parallel mode in case of it was chosen ..
	// omp_set_num_threads(NUM_OF_THREADS);
	// solve_jacobi_parallel(matrix, matrix_size, right_hand_side);
	// printf("Elapsed time: %f ms\n", float(clock() - parallel_starting_time));

	// Cleaning the chaos
	delete_matrix(matrix, N);
	delete[] right_hand_side;
	
}

bool check_diagoanally_dominant_parallel(float** matrix, int matrix_size){
	// This is to validate that all the rows applies the rule .. 
	int check_count = 0;
	#pragma omp parallel 
	{
		// For each row
		// Each thread will be assigned to run on a row.
		#pragma omp for schedule (guided, 1)
		for (int i = 0; i < matrix_size; i++){
			float row_sum = 0;
			// Summing the other row elements .. 
			for (int j = 0; j < matrix_size; j++) {
				if (j != i) row_sum += abs(matrix[i][j]);
			}

			if (abs(matrix[i][i]) >= row_sum){
				#pragma omp atomic 
				check_count++;
			}
		}
	}
	return check_count == matrix_size;
}

bool check_diagoanally_dominant_sequential(float** matrix, int matrix_size){
	int check_count = 0;
	// For each row ..
	for (int i = 0; i < matrix_size; i++) {
		float row_sum = 0;
		// Summing the other row elements .. 
		for (int j = 0; j < matrix_size; j++) {
			if (j != i) row_sum += abs(matrix[i][j]);
		}

		if (abs(matrix[i][i]) >= row_sum) {
			check_count++;
		}
	}
	return check_count == matrix_size;
}

void solve_jacobi_sequential(float* matrix, int n, float* right_hand_side) {
	float* solution = new float[n];
	float* last_iteration = new float[n];
	
	for(int l = 0; l<N; l++){
		solution[l] = 0;
		last_iteration[l] = 0;
	}

	// Just for initialization ..
	printf("Iterations:----------------------------------------- \n");
	init_array_sequential(solution, n);
	
	for (int l = 0; l < IT; l++){
		for (int i = 0; i < n; i++) {
			float sigma_value = 0;
			//printf("i: %d, x : %f\n", i,last_iteration[i]);
			for (int j = 0; j < n; j++) {
				if (i != j) {
					sigma_value += matrix[i+j*n] * last_iteration[j];
					//printf("sigma_value %f\n",sigma_value);
				}
			}
			solution[i] = (right_hand_side[i] - sigma_value) / matrix[i+i*n];
			//printf(" id: %d, x_res : %f\n", i,solution[i]);

		}
		last_iteration = clone_array_sequential(solution, n);
		swap(last_iteration, solution);
		printf("Iteration #%d: ", l+1);
		for (int l = 0; l < n; l++) {
			printf("%f ", solution[l]);
		}
		printf("\n");
	}
}

void solve_jacobi_parallel(float** matrix, int matrix_size, float* right_hand_side) {
	float* solution = new float[matrix_size];
	float* last_iteration = new float[matrix_size];

	// Just for initialization ..
	printf("Iterations:--------------------------------------------------\n");
	init_array_parallel(solution, matrix_size); // dump the array with zeroes

	// NOTE: we don't need to parallelize this as the iterations are dependent. However, we may parallelize the inner processes 
	for (int i = 0; i < ITERATION_LIMIT; i++){
		// Make a deep copy to a temp array to compare it with the resulted vector later
		last_iteration = clone_array_parallel(solution, matrix_size);
		
		// Each thread is assigned to a row to compute the corresponding solution element
		#pragma omp parallel for schedule(dynamic, 1)
		for (int j = 0; j < matrix_size; j++){
			float sigma_value = 0;
			for (int k = 0; k < matrix_size; k++){
				if (j != k) {
					sigma_value += matrix[j][k] * solution[k];
				}
			}
			solution[j] = (right_hand_side[j] - sigma_value) / matrix[j][j];
		}

		// Checking for the stopping condition ...
		int stopping_count = 0;
		#pragma omp parallel for schedule(dynamic, 1) 
		for (int s = 0; s < matrix_size; s++) {
			if (abs(last_iteration[s] - solution[s]) <= EPSILON) {
				#pragma atomic
				stopping_count++;
			}
		}

		if (stopping_count == matrix_size) break;

		printf("Iteration #%d: ", i+1);
		for (int l = 0; l < matrix_size; l++) {
			printf("%f ", solution[l]);
		}
		printf("\n");
	}
}

void init_array_sequential(float array[], int array_size){
	for (int i = 0; i < array_size; i++) {
		array[i] = 0;
	}
}

float* clone_array_sequential(float array[], int array_length){
	float* output = new float[array_length];
	for (int i = 0; i < array_length; i++) {
		output[i] = array[i];
	}
	return output;
}

void init_array_parallel(float array[], int array_size){
	#pragma omp parallel for schedule (dynamic, 1)
	for (int i = 0; i < array_size; i++) {
		array[i] = 0;
	}
}

float* clone_array_parallel(float array[], int array_length){
	float* output = new float[array_length];
	#pragma omp parallel for schedule (dynamic, 1)
	for (int i = 0; i < array_length; i++) {
		output[i] = array[i];
	}
	return output;
}

void delete_matrix(float* matrix, int matrix_size) {
	delete[] matrix;
}

