#include "b16.cuh"

//////////////////////////////
//////////////////////////////

extern "C" __global__ void nb_1_multi(const int* x, const float* y, float* z, int size, int n_feat, int n_classes) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_classes; j++) {
            for (int q = 0; q < n_feat; q++) {
                z[i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
            }
        }
    }
}

extern "C" __global__ void nb_2_multi(const float* x, float* y, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        float curr_max = x[i * n_col_x];
        for (int j = 0; j < n_col_x; j++) {
            curr_max = fmaxf(curr_max, x[i * n_col_x + j]);
        }
        y[i] = curr_max;
    }
}

extern "C" __global__ void nb_3_multi(const float* x, const float* y, float* z, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < n_col_x; j++) {
            sum += expf(x[i * n_col_x + j] - y[i]);
        }
        z[i] = logf(sum) + y[i];
    }
}

extern "C" __global__ void nb_4_multi(float* x, float* y, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] = expf(x[i * n_col_x + j] - y[i]);
        }
    }
}

__inline__ __device__ float warp_reduce_multi(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void rr_1_0_multi(const int* x, float* y, float* z, int n_row_x, int n_col_x) {
    int warp_size = 32;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x; j += blockDim.x * gridDim.x) {
        // Compute mean and variance;
        float feature_mean = float(0);
        float sum_sq = float(0);
        for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < n_row_x; i += blockDim.y * gridDim.y) {
            float x_tmp = x[j * n_row_x + i];
            feature_mean += x_tmp;
            sum_sq += x_tmp * x_tmp;
        }
        feature_mean = warp_reduce_multi(feature_mean);  // Obtain the sum of values in the current warp;
        sum_sq = warp_reduce_multi(sum_sq);              // Obtain the sum of values in the current warp;
        if (!(threadIdx.y % warp_size)) {
            atomicAdd(y + j, feature_mean);
            atomicAdd(z + j, sum_sq);
        }
    }
}

extern "C" __global__ void rr_1_1_multi(const int* x, float* y, const float* mean, const float* std, int n_row_x, int n_col_x) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x; j += blockDim.x * gridDim.x) {
        float mean_tmp = mean[j] / n_row_x;
        float std_tmp = sqrtf(std[j] / n_row_x - mean_tmp * mean_tmp);

        for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < n_row_x; i += blockDim.y * gridDim.y) {
            y[j * n_row_x + i] = ((float)x[j * n_row_x + i] - mean_tmp) / std_tmp;
        }
    }
}

extern "C" __global__ void rr_1_multi(const int* x, float* y, int n_row_x, int n_col_x) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x; j += blockDim.x * gridDim.x) {
        float feature_mean = 0;
        float sum_sq = 0;
        // Compute mean and variance;
        for (int i = 0; i < n_row_x; i++) {
            float x_tmp = x[j * n_row_x + i];
            feature_mean += x_tmp;
            sum_sq += x_tmp * x_tmp;
        }
        feature_mean /= n_row_x;
        float std = sqrtf(sum_sq / n_row_x - feature_mean * feature_mean);

        // Update values;
        for (int i = 0; i < n_row_x; i++) {
            y[j * n_row_x + i] = (x[j * n_row_x + i] - feature_mean) / std;
        }
    }
}

extern "C" __global__ void rr_2_multi(const float* x, const float* y, float* z, int size, int n_feat, int n_classes) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_classes; j++) {
            for (int q = 0; q < n_feat; q++) {
                z[i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
            }
        }
    }
}

extern "C" __global__ void rr_3_multi(float* x, const float* y, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] += y[j];
        }
    }
}

extern "C" __global__ void softmax_multi(float* x, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        float row_exp_sum = 0;
        for (int j = 0; j < n_col_x; j++) {
            row_exp_sum += expf(x[i * n_col_x + j]);
        }
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] = expf(x[i * n_col_x + j]) / row_exp_sum;
        }
    }
}

extern "C" __global__ void argmax_multi(const float* x, const float* y, int* z, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        int curr_best_index = 0;
        float curr_best = x[i * n_col_x] + y[i * n_col_x];
        for (int j = 0; j < n_col_x; j++) {
            float curr = x[i * n_col_x + j] + y[i * n_col_x + j];
            if (curr > curr_best) {
                curr_best = curr;
                curr_best_index = j;
            }
        }
        z[i] = curr_best_index;
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark16::alloc() {


    cudaSetDevice(0);            // Set device 0 as current
    err = cudaMallocManaged(&x0, sizeof(int) * N * num_features);
    err = cudaMallocManaged(&z, sizeof(float) * N * num_features);
    err = cudaMallocManaged(&ridge_coeff, sizeof(float) * num_classes * num_features);
    err = cudaMallocManaged(&r2, sizeof(float) * N * num_classes);
    err = cudaMallocManaged(&ridge_intercept, sizeof(float) * num_classes);
    err = cudaMallocManaged(&r, sizeof(int) * N);
    //nb_class_log_prior not used in kernels
    err = cudaMallocManaged(&nb_class_log_prior, sizeof(float) * num_classes);
    err = cudaStreamCreate(&s1);

    cudaSetDevice(1);            // Set device 1 as current
    err = cudaMallocManaged(&nb_feat_log_prob, sizeof(float) * num_classes * num_features);
    err = cudaMallocManaged(&x1, sizeof(int) * N * num_features);//
    err = cudaMallocManaged(&nb_amax, sizeof(float) * N);
    err = cudaMallocManaged(&r1, sizeof(float) * N * num_classes);
    err = cudaMallocManaged(&nb_l, sizeof(float) * N);
    err = cudaStreamCreate(&s2);

}

void Benchmark16::init() {
    int max_occurrence_of_ngram = 10;

    //init for device 0
    cudaSetDevice(0);            // Set device 0 as current


    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < num_features; j++) {
            ridge_coeff[i * num_features + j] = (float)(rand()) / (float)(RAND_MAX);
        }
        nb_class_log_prior[i] = (float)(rand()) / (float)(RAND_MAX);
        ridge_intercept[i] = (float)(rand()) / (float)(RAND_MAX);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < num_features; j++) {
            x0[i * num_features + j] = rand() % max_occurrence_of_ngram;

        }
        for (int j = 0; j < num_classes; j++) {
            r1[i * num_classes + j] = nb_class_log_prior[j];
        }
    }

    //init for device 1
    cudaSetDevice(1);            // Set device 1 as current

    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < num_features; j++) {
            nb_feat_log_prob[i * num_features + j] = (float)(rand()) / (float)(RAND_MAX);
        }
        nb_class_log_prior[i] = (float)(rand()) / (float)(RAND_MAX);
        ridge_intercept[i] = (float)(rand()) / (float)(RAND_MAX);
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < num_features; j++) {
            x1[i * num_features + j] = rand() % max_occurrence_of_ngram;

        }
        for (int j = 0; j < num_classes; j++) {
            r2[i * num_classes + j] = 0;
        }
    }
}

void Benchmark16::reset() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < num_classes; j++) {
            r1[i * num_classes + j] = nb_class_log_prior[j];
            r2[i * num_classes + j] = 0;
        }
        // r1_mean[i] = 0;
        // r1_std[i] = 0;
    }
}

void Benchmark16::execute_sync(int iter) {
//     rr_1_multi<<<num_blocks, block_size_1d>>>(x, z, N, num_features);
//     // dim3 num_blocks_2d(8, 8);
//     // dim3 block_size_1d_2d(1, 32);
//     // rr_1_0<<<num_blocks_2d, block_size_1d_2d>>>(x, r1_mean, r1_std, N, num_features);
//     // cudaDeviceSynchronize();
//     // rr_1_1<<<num_blocks_2d, block_size_1d_2d>>>(x, z, r1_mean, r1_std, N, num_features);
//     cudaDeviceSynchronize();

//     // auto e1 = clock_type::now();
//     // auto rr1time = chrono::duration_cast<chrono::microseconds>(e1 - start).count();
//     // if (debug) std::cout << " rr1=" << (float) rr1time / 1000 << " ms" << std::endl;

//     nb_1_multi<<<num_blocks, block_size_1d>>>(x, nb_feat_log_prob, r1, N, num_features, num_classes);
//     cudaDeviceSynchronize();

//     rr_2_multi<<<num_blocks, block_size_1d>>>(z, ridge_coeff, r2, N, num_features, num_classes);
//     cudaDeviceSynchronize();

//     nb_2_multi<<<num_blocks, block_size_1d>>>(r1, nb_amax, N, num_classes);
//     cudaDeviceSynchronize();

//     nb_3_multi<<<num_blocks, block_size_1d>>>(r1, nb_amax, nb_l, N, num_classes);
//     cudaDeviceSynchronize();

//     rr_3_multi<<<num_blocks, block_size_1d>>>(r2, ridge_intercept, N, num_classes);
//     cudaDeviceSynchronize();

//     nb_4_multi<<<num_blocks, block_size_1d>>>(r1, nb_l, N, num_classes);
//     cudaDeviceSynchronize();

//     softmax_multi<<<num_blocks, block_size_1d>>>(r1, N, num_classes);
//     cudaDeviceSynchronize();

//     softmax_multi<<<num_blocks, block_size_1d>>>(r2, N, num_classes);
//     cudaDeviceSynchronize();

//     argmax_multi<<<num_blocks, block_size_1d>>>(r1, r2, r, N, num_classes);
//     cudaDeviceSynchronize();
}

void Benchmark16::execute_async(int iter) {
    cudaSetDevice(0);            // Set device 0 as current
    cudaStreamAttachMemAsync(s1, z, 0);
    cudaStreamAttachMemAsync(s1, ridge_coeff, 0);
    cudaStreamAttachMemAsync(s1, r2, 0);
    cudaStreamAttachMemAsync(s1, ridge_intercept, 0);

    //changes in z
    rr_1_multi<<<num_blocks, block_size_1d, 0, s1>>>(x0, z, N, num_features);
    //changes in r2
    rr_2_multi<<<num_blocks, block_size_1d, 0, s1>>>(z, ridge_coeff, r2, N, num_features, num_classes);
    //changes r2
    rr_3_multi<<<num_blocks, block_size_1d, 0, s1>>>(r2, ridge_intercept, N, num_classes);
    //change r2
    softmax_multi<<<num_blocks, block_size_1d, 0, s1>>>(r2, N, num_classes);

    cudaSetDevice(1);            // Set device 0 as current

    cudaStreamAttachMemAsync(s2, nb_feat_log_prob, 0);
    cudaStreamAttachMemAsync(s2, r1, 0);
    cudaStreamAttachMemAsync(s2, nb_amax, 0);
    cudaStreamAttachMemAsync(s2, nb_l, 0);


    //changes in r1
    nb_1_multi<<<num_blocks, block_size_1d, 0, s2>>>(x1, nb_feat_log_prob, r1, N, num_features, num_classes);
    //changes in nb_max
    nb_2_multi<<<num_blocks, block_size_1d, 0, s2>>>(r1, nb_amax, N, num_classes);
    //changes nb_l
    nb_3_multi<<<num_blocks, block_size_1d, 0, s2>>>(r1, nb_amax, nb_l, N, num_classes);
    //changes r1
    nb_4_multi<<<num_blocks, block_size_1d, 0, s2>>>(r1, nb_l, N, num_classes);
    //change r1
    softmax_multi<<<num_blocks, block_size_1d, 0, s2>>>(r1, N, num_classes);


    // Stream 1 waits stream 2;
    cudaEvent_t e1;
    cudaEventCreate(&e1);
    cudaEventRecord(e1, s2);
    cudaStreamWaitEvent(s1, e1, 0);


    cudaSetDevice(0);            // Set device 0 as current
    //change r
    argmax_multi<<<num_blocks, block_size_1d, 0, s1>>>(r1, r2, r, N, num_classes);
    cudaDeviceSynchronize();
}

void Benchmark16::execute_cudagraph(int iter) {}

void Benchmark16::execute_cudagraph_manual(int iter) {}

std::string Benchmark16::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(r[0]);
    } else {
        std::string res = "[";
        for (int j = 0; j < 10; j++) {
            res += std::to_string(r[j]) + ", ";
        }
        return res + ", ...]";
    }
}