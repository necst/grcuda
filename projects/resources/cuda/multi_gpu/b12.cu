// Copyright (c) 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include "b12.cuh"

extern "C" __global__ void subtract(float* v1, const float* v2, const float alpha, int N, int offset) {
    int init = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = init; i < N; i += stride){
        v1[i] -= alpha * v2[i + offset];
    }
}

__global__ void copy_partition_to_vec(const float *vec_in, float *vec_out, const int N, const int offset_out, const int offset_in){
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = init; i < N; i += stride){
        vec_out[i + offset_out] = vec_in[i + offset_in];
    }
}

__global__ void normalize(const float *d_v_in, const float denominator, float *d_v_out, int N) {
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < N; i += stride) {
        d_v_out[i] = d_v_in[i] * denominator;
    }
}

__global__ void spmv(const int *x, const int *y, const float *val, const float *v_in, float *v_out, int num_nnz) {
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < num_nnz; i += stride) {
        v_out[y[i]] += v_in[x[i]] * val[i];
    }
}

__global__ void dot_product_stage_one(const float* v1, const float* v2, float* temporaryOutputValues, int N, int offset) {
    __shared__ float cache[DOT_PRODUCT_THREADS_PER_BLOCK];
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    int stride = blockDim.x * gridDim.x;
    float temp = 0;
    for(int i = threadId; i < N; i += stride){
        temp += v1[i] * v2[i + offset];
    }

    cache[cacheIdx] = temp;

    __syncthreads();

    for(int i = blockDim.x >> 1; i != 0; i >>= 1){
        if(cacheIdx < i){
            cache[cacheIdx] += cache[cacheIdx + 1];
        }
        __syncthreads();
    }

    if (cacheIdx == 0){
        temporaryOutputValues[blockIdx.x] = cache[0];
    }
}

__global__ void dot_product_stage_two(const float *temporary_results, float *result) {

    float acc = temporary_results[threadIdx.x];
    for(int i = 16; i > 0; i >>= 1){
        acc += __shfl_down_sync(0xffffffff, acc, i);
        __syncthreads();
    }

    __syncthreads();

    if(threadIdx.x == 0) *result = acc;
}

/**
 *  -alpha,
                    this->vec_in[p_idx], -> x ->offset this->offsets[p_idx],
                    this->spmv_vec_out[p_idx], -> b
                    -beta,
                    this->lanczos_vectors[p_idx], -> c
                    this->vec_next[p_idx], -> out
                    this->coo_partitions[p_idx].N,
                    this->offsets[p_idx],
                    this->coo_partitions[p_idx].N * (i - 1)
 */

__global__ void axpb_xtended(const float alpha, const float *x, const float *b, const float beta, const float *c, float *out, const int N, const int offset_x, const int offset_c) {
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = init; i < N; i += stride) {
        out[i] = alpha * x[i + offset_x] + b[i] + beta * c[i + offset_c];
    }
}


void Benchmark12::alloc_vectors() {
    for (const auto& partition: this->coo_partitions){
        f32 *tmp_vec_in, *tmp_spmv_out, *tmp_intermediate_dot_product_values;
        f32 *tmp_vec_next, *tmp_lanczos_vectors, *tmp_normalized_out;

        cudaMallocManaged(&tmp_vec_in, sizeof(f32) * this->matrix.N);
        cudaMallocManaged(&tmp_spmv_out, sizeof(f32) * partition.N);
        cudaMallocManaged(&tmp_intermediate_dot_product_values, sizeof(f32) * 32);
        cudaMallocManaged(&tmp_vec_next, sizeof(f32) * partition.N);
        cudaMallocManaged(&tmp_lanczos_vectors, sizeof(f32) * this->num_eigencomponents * partition.N);
        cudaMallocManaged(&tmp_normalized_out, sizeof(f32) * partition.N);

        this->vec_in.push_back(tmp_vec_in);
        this->spmv_vec_out.push_back(tmp_spmv_out);
        this->intermediate_dot_product_values.push_back(tmp_intermediate_dot_product_values);
        this->vec_next.push_back(tmp_vec_next);
        this->lanczos_vectors.push_back(tmp_lanczos_vectors);
        this->normalized_out.push_back(tmp_normalized_out);
    }

    cudaMallocManaged(&alpha_intermediate, sizeof(f32) * this->num_gpus);
    cudaMallocManaged(&beta_intermediate, sizeof(f32) * this->num_gpus);
}

void Benchmark12::alloc_coo_partitions() {

    const u32 nnz_per_partition = u32((this->matrix.nnz + this->num_gpus) / this->num_gpus);
    u32 from_index = 0;
    u32 to_index = nnz_per_partition;
    u32 index_value = this->matrix.y[to_index];

    for(u32 i = 0; i < this->num_gpus - 1; ++i){
        std::cout << "partition " << i << std::endl;
        while(index_value == this->matrix.y[to_index]) {
            to_index++;
        }
        const u32 offset = (from_index == 0) ? from_index : (this->matrix.y[from_index] - 1);
        auto coo_partition = *(this->assign_partition(from_index, to_index, offset));
        this->coo_partitions.push_back(coo_partition);

        from_index = to_index;
        to_index += nnz_per_partition;
        index_value = this->matrix.y[to_index];
    }

    const u32 offset = this->matrix.y[from_index];
    auto coo_partition = *(this->assign_partition(from_index, this->matrix.nnz, offset));
    this->coo_partitions.push_back(coo_partition);
}

coo_matrix_t *Benchmark12::assign_partition(u32 from_index, u32 to_index, u32 offset) {
    //std::cout << "from " << from_index << " to " << to_index << " offset " << offset << std::endl;
    coo_matrix_t *coo_partition;
    cudaMallocManaged(&coo_partition, sizeof(coo_matrix_t) );
    coo_partition->begin = from_index;
    coo_partition->end = to_index;
    //std::cout << "Malloc" << std::endl;
    cudaMallocManaged(&coo_partition->x, sizeof(u32) * (to_index - from_index));
    cudaMallocManaged(&coo_partition->y, sizeof(u32) * (to_index - from_index));
    cudaMallocManaged(&coo_partition->val, sizeof(u32) * (to_index - from_index));

    //std::cout << "copiying" << std::endl;
    u32 j = 0;
    for(u32 i = from_index; i < to_index; ++i,++j){
        coo_partition->x[j]   = this->matrix.x[i];
        coo_partition->y[j]   = this->matrix.y[i] - offset;
        coo_partition->val[j] = this->matrix.val[i];
    }

    coo_partition->N = coo_partition->y[to_index - from_index - 1];
    coo_partition->nnz = to_index - from_index;
    std::cout << "done" << std::endl;
    return coo_partition;
}

void Benchmark12::create_random_matrix(bool normalize = true) {
    u32 total_nnz = RANDOM_MATRIX_AVG_NNZ_PER_ROW * RANDOM_MATRIX_NUM_ROWS;
    i32 *x        = (i32*) std::malloc(total_nnz * sizeof(i32));
    i32 *y        = (i32*) std::malloc(total_nnz * sizeof(i32));
    f32 *val      = (f32*) std::malloc(total_nnz * sizeof(f32));

    f32 value_to_set = normalize ? (1.0f / RANDOM_MATRIX_NUM_ROWS) : 1.0f;

    for(u32 i = 0; i < total_nnz; ++i)
        val[i] = value_to_set;

    auto random_node = [&](){
        return std::rand() % RANDOM_MATRIX_NUM_ROWS;
    };


    std::generate(x, x + total_nnz, random_node);
    std::generate(y, y + total_nnz, random_node);

    std::sort(y, y + total_nnz);

    this->matrix.x = x;
    this->matrix.y = y;
    this->matrix.val = val;
    this->matrix.begin = 0;
    this->matrix.end = total_nnz;
    this->matrix.N = RANDOM_MATRIX_NUM_ROWS;
    this->matrix.nnz = total_nnz;

}

void Benchmark12::alloc() {

    if(this->matrix_path.empty())
        this->create_random_matrix();

    this->create_streams();
    this->alloc_coo_partitions();
    this->alloc_vectors();

    // Create offsets
    this->offsets.push_back(0);
    for(u32 i = 1; i < this->num_gpus; ++i)
        this->offsets.push_back(this->coo_partitions[i].N - this->offsets[i - 1]);

}

void Benchmark12::reset() {
    // Just call init, it resets all the necessary vectors;
    this->init();
}

void Benchmark12::create_streams() {

    for(u32 i = 0; i < this->num_gpus; ++i){
        cudaStream_t *stream = (cudaStream_t*) std::malloc(sizeof(cudaStream_t));
;        cudaStreamCreate(stream);
        this->streams.push_back(*stream);
    }

}

template <typename Function>
void Benchmark12::launch_multi_kernel(Function kernel_launch_function) {


    for(u32 i = 0; i < this->num_gpus; ++i) {
        cudaSetDevice(i);
        cudaStream_t stream = policy == Policy::Sync ? nullptr : streams[i];
        kernel_launch_function(i, stream);

        if(policy == Policy::Sync)
            cudaDeviceSynchronize();
    }

}

void Benchmark12::execute(i32 iter) {

    std::cout << this->matrix << std::endl;

    for(const auto& p: this->coo_partitions)
        std::cout << p << std::endl;

    f32 alpha = 0.0f;
    f32 beta  = 0.0f;
    f32* alpha_storage_host = (f32 *) std::malloc(this->num_gpus * sizeof(f32));
    f32* beta_storage_host = (f32 *) std::malloc(this->num_gpus * sizeof(f32));

    if(this->debug)
        std::cout << "[LANCZOS - Sync] Iteration " << iter << std::endl;
    std::cout << "spmv" << std::endl;
    this->launch_multi_kernel([this](u32 p_idx, cudaStream_t stream){
        spmv<<<this->num_blocks, this->block_size, 0, stream>>>(
                this->coo_partitions[p_idx].x,
                this->coo_partitions[p_idx].y,
                this->coo_partitions[p_idx].val,
                this->vec_in[p_idx],
                this->spmv_vec_out[p_idx],
                this->coo_partitions[p_idx].nnz
        );
    });

    cudaDeviceSynchronize();
    std::cout << "dp" << std::endl;

    this->launch_multi_kernel([this](u32 p_idx, cudaStream_t stream){
        dot_product_stage_one<<<DOT_PRODUCT_NUM_BLOCKS, this->block_size * (this->num_blocks / DOT_PRODUCT_NUM_BLOCKS), 0, stream>>>(
            this->vec_in[p_idx],
            this->spmv_vec_out[p_idx],
            this->intermediate_dot_product_values[p_idx],
            this->coo_partitions[p_idx].N,
            this->offsets[p_idx]
        );
        dot_product_stage_two<<<1, 32, 0, stream>>>(
            this->intermediate_dot_product_values[p_idx],
            &this->alpha_intermediate[p_idx]
        );
    });

    cudaDeviceSynchronize();
    std::cout << "alpha" << std::endl;
    cudaMemcpy(alpha_storage_host, this->alpha_intermediate, this->num_gpus, cudaMemcpyDeviceToHost);
    alpha = std::accumulate(alpha_storage_host, alpha_storage_host + this->num_gpus, 0.0f);
    tridiagonal_matrix.push_back(alpha);
    std::cout << "axpb" << std::endl;

    this->launch_multi_kernel([this, alpha](u32 p_idx, cudaStream_t stream){
        axpb_xtended<<<this->num_blocks, this->block_size, 0, stream>>>(
            -alpha,
            this->vec_in[p_idx],
            this->spmv_vec_out[p_idx],
            0,
            this->vec_in[p_idx],
            this->vec_next[p_idx],
            this->coo_partitions[p_idx].N,
            this->offsets[p_idx],
            0
        );
    });

    for(u32 i = 0; i < this->num_eigencomponents; ++i){
        std::cout << "eigen " << i << std::endl;
        std::cout << "normalize - dp" << std::endl;

        this->launch_multi_kernel([this](u32 p_idx, cudaStream_t stream){
            dot_product_stage_one<<<DOT_PRODUCT_NUM_BLOCKS, this->block_size * (this->num_blocks / DOT_PRODUCT_NUM_BLOCKS), 0, stream>>>(
                    this->vec_next[p_idx],
                    this->vec_next[p_idx],
                    this->intermediate_dot_product_values[p_idx],
                    this->coo_partitions[p_idx].N,
                    0
            );
            dot_product_stage_two<<<1, 32, 0, stream>>>(
                    this->intermediate_dot_product_values[p_idx],
                    &this->beta_intermediate[p_idx]
            );
        });

        std::cout << "beta" << std::endl;
        cudaMemcpy(beta_storage_host, this->beta_intermediate, this->num_gpus, cudaMemcpyDeviceToHost);
        beta = std::accumulate(beta_storage_host, beta_storage_host + this->num_gpus, 0.0f);
        tridiagonal_matrix.push_back(beta);

        std::cout << "normalize" << std::endl;

        this->launch_multi_kernel([this, beta](u32 p_idx, cudaStream_t stream){
                normalize<<<this->num_blocks, this->block_size, 0, stream>>>(
                        this->vec_next[p_idx],
                        1.0f / beta,
                        this->normalized_out[p_idx],
                        this->coo_partitions[p_idx].N
                );
            }
        );
        std::cout << "copy p t v" << std::endl;
        this->launch_multi_kernel([this, i](u32 p_idx, cudaStream_t stream){
            copy_partition_to_vec<<<this->num_blocks, this->block_size>>>(
                        this->vec_in[p_idx],
                        this->lanczos_vectors[p_idx],
                        this->coo_partitions[p_idx].N,
                        this->coo_partitions[p_idx].N * (i - 1),
                        this->offsets[p_idx]
            );
        });


        for(u32 j = 0; j < this->num_gpus; ++j){
            std::cout << "copy p t v " << j << std::endl;

            this->launch_multi_kernel([this, i](u32 p_idx, cudaStream_t stream){
                copy_partition_to_vec<<<this->num_blocks, this->block_size>>>(
                        this->normalized_out[p_idx],
                        this->vec_in[p_idx],
                        this->coo_partitions[p_idx].N,
                        offsets[p_idx],
                        0
                );
            });

            auto first = this->vec_in.front();
            this->vec_in.erase(this->vec_in.begin());
            this->vec_in.push_back(first);
        }


        std::cout << "spmv" << std::endl;

        this->launch_multi_kernel([this](u32 p_idx, cudaStream_t stream){
            spmv<<<this->num_blocks, this->block_size, 0, stream>>>(
                    this->coo_partitions[p_idx].x,
                    this->coo_partitions[p_idx].y,
                    this->coo_partitions[p_idx].val,
                    this->vec_in[p_idx],
                    this->spmv_vec_out[p_idx],
                    this->coo_partitions[p_idx].nnz
            );
        });

        std::cout << "dp" << std::endl;

        this->launch_multi_kernel([this](u32 p_idx, cudaStream_t stream){
            dot_product_stage_one<<<DOT_PRODUCT_NUM_BLOCKS, this->block_size * (this->num_blocks / DOT_PRODUCT_NUM_BLOCKS), 0, stream>>>(
                    this->vec_in[p_idx],
                    this->spmv_vec_out[p_idx],
                    this->intermediate_dot_product_values[p_idx],
                    this->coo_partitions[p_idx].N,
                    this->offsets[p_idx]
            );
            dot_product_stage_two<<<1, 32, 0, stream>>>(
                    this->intermediate_dot_product_values[p_idx],
                    &this->alpha_intermediate[p_idx]
            );
        });
        std::cout << "alpha" << std::endl;

        cudaMemcpy(alpha_storage_host, this->alpha_intermediate, this->num_gpus, cudaMemcpyDeviceToHost);
        alpha = std::accumulate(alpha_storage_host, alpha_storage_host + this->num_gpus, 0.0f);
        tridiagonal_matrix.push_back(alpha);
        std::cout << "axpb" << std::endl;

        this->launch_multi_kernel([this, alpha, beta, i](u32 p_idx, cudaStream_t stream){

            std::cout << "this->coo_partitions[p_idx].N  = " << this->coo_partitions[p_idx].N  << " this->coo_partitions[p_idx].N * (i - 1) = " << this->coo_partitions[p_idx].N * (i - 1) << std::endl;
            std::cout << "this->offsets[p_idx] = " << this->offsets[p_idx] << std::endl;
            axpb_xtended<<<this->num_blocks, this->block_size, 0, stream>>>(
                    -alpha,
                    this->vec_in[p_idx],
                    this->spmv_vec_out[p_idx],
                    -beta,
                    this->lanczos_vectors[p_idx],
                    this->vec_next[p_idx],
                    this->coo_partitions[p_idx].N,
                    this->offsets[p_idx],
                    this->coo_partitions[p_idx].N * (i - 1)
            );
        });


        if(this->reorthogonalize){

            for(u32 j = 0; j < i; ++j){
                this->launch_multi_kernel([this, j](u32 p_idx, cudaStream_t stream){
                    dot_product_stage_one<<<DOT_PRODUCT_NUM_BLOCKS, this->block_size * (this->num_blocks / DOT_PRODUCT_NUM_BLOCKS), 0, stream>>>(
                            this->vec_next[p_idx],
                            this->lanczos_vectors[p_idx],
                            this->intermediate_dot_product_values[p_idx],
                            this->coo_partitions[p_idx].N,
                            this->offsets[p_idx] * j
                    );
                    dot_product_stage_two<<<1, 32, 0, stream>>>(
                            this->intermediate_dot_product_values[p_idx],
                            &this->alpha_intermediate[p_idx]
                    );
                });

                alpha = std::accumulate(this->alpha_intermediate, this->alpha_intermediate + this->num_gpus, 0.0f);

                this->launch_multi_kernel([this, alpha](u32 p_idx, cudaStream_t stream){
                    subtract<<<this->num_blocks, this->block_size, 0, stream>>>(
                            this->vec_next[p_idx],
                            this->lanczos_vectors[p_idx],
                            alpha,
                            this->coo_partitions[p_idx].N,
                            this->coo_partitions[p_idx].N
                    );
                });

            }

        }

        cudaDeviceSynchronize();

    }

}

void Benchmark12::execute_sync(i32 iter) {
    assert(this->policy == Policy::Sync);
    this->execute(iter);
}

void Benchmark12::execute_async(int iter) {
    assert(this->policy == Policy::Async);
    this->execute(iter);
}

std::string Benchmark12::print_result(bool short_form = false) {
    std::string str;
    for(u32 i = 0; i < tridiagonal_matrix.size(); ++i){
        str += "" +  std::to_string(i) + ") " + std::to_string(tridiagonal_matrix[i]) + "\n";// std::endl;
    }
    return str;
}

void Benchmark12::init() {
    // Initialize vec_in[0]
    std::generate(this->vec_in[0], this->vec_in[0] + this->matrix.N, std::rand);
    f32 norm = std::accumulate(this->vec_in[0], this->vec_in[0] + this->matrix.N, 0.0f, [](f32 acc, f32 cur){
        return acc + cur * cur;
    });

    // Normalize it
    for(u32 i = 0; i < this->matrix.N; ++i){
        this->vec_in[0][i] /= norm;
    }

    // copy it to the other vectors
    for(u32 i = 1; i < this->num_gpus; ++i){
        cudaMemcpy(this->vec_in[i], this->vec_in[0], this->matrix.N, cudaMemcpyHostToHost);
    }

    // Initialize the other vectors that get
    // both read and written in a single computation
    for(u32 i = 0; i < this->num_gpus; ++i){
        const auto& partition = this->coo_partitions[i];

        for(u32 j = 0; j < partition.N; ++j){
            this->spmv_vec_out[i][j]    = 0.0f;
            this->vec_next[i][j]        = 0.0f;
            this->normalized_out[i][j]  = 0.0f;
        }

    }

}

void Benchmark12::execute_cudagraph(int iter) {
    throw new std::runtime_error("Benchmark12::execute_cudagraph not implemented");
}

void Benchmark12::execute_cudagraph_manual(int iter) {
    throw new std::runtime_error("Benchmark12::execute_cudagraph_manual not implemented");
}

void Benchmark12::execute_cudagraph_single(int iter) {
    throw new std::runtime_error("Benchmark12::execute_cudagraph_single not implemented");
}

std::ostream &operator<<(std::ostream &os, const coo_matrix_t &matrix) {
    os << "x: " << matrix.x << " y: " << matrix.y << " val: " << matrix.val << " begin: " << matrix.begin << " end: "
       << matrix.end << " N: " << matrix.N << " nnz: " << matrix.nnz;
    return os;
}
