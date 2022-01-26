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


#include <sstream>
#include "b12.cuh"

void read_matrix(const std::string file_path, i32 &N, i32 &M, i32 &NNZ, std::vector<i32> &x, std::vector<i32> &y,
                 std::vector<f32> &val) {
    std::ifstream input(file_path);
    std::string cur_line;
    int cur_x, cur_y;

    while (std::getline(input, cur_line)) {
        if (cur_line[0] != '%') {

            std::istringstream iss(cur_line);
            iss >> N >> M >> NNZ;
            break;
        }
    }


    while (std::getline(input, cur_line)) {
        std::istringstream iss(cur_line);

        iss >> cur_x >> cur_y;
        cur_x--;
        cur_y--;
        x.push_back(cur_x);
        y.push_back(cur_y);
        val.push_back(1.0);
    }

}

/**
 * From https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define CUDA_CHECK_ERROR(kernel_ret_code) { gpuAssert((kernel_ret_code), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        //if (abort) exit(code);
    }
}

__global__ void subtract(float *v1, const float *v2, const float alpha, int N, int offset) {
    int init = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < N; i += stride) {
        v1[i] -= alpha * v2[i + offset];
    }
}

__global__ void
copy_partition_to_vec(const float *vec_in, float *vec_out, const int N, const int offset_out, const int offset_in) {
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = init; i < N; i += stride) {
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

    //printf("BlockID: %d, ThreadID: %d. Init = %d, Stride = %d\n", blockIdx.x, threadIdx.x, init, stride);

    for (int i = init; i < num_nnz; i += stride) {
        //printf("v_out[%d] += v_in[%d] * val[%d]\n", y[i], x[i], i);
        v_out[y[i]] += v_in[x[i]] * val[i];
    }
}


__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// z = <x, x>;
extern "C" __global__ void l2_norm_b12(const float *x, float *z, int N, int offset) {
    int warp_size = 32;
    float sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float x_tmp = x[i + offset];
        sum += x_tmp * x_tmp;
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}

__global__ void dot_product(const float *x, const float *y, float *z, const int N, const int offset) {

    static constexpr int WARP_SIZE = 32;
    float sum = 0;
    const u32 begin = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 stride = blockDim.x * gridDim.x;
    for (u32 i = begin; i < N; i += stride) {
        sum += x[i] * y[i + offset];
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) { // Same as (threadIdx.x % WARP_SIZE) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
    }
}

__global__ void set(float *v_in, float value, const int N) {
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (u32 i = init; i < N; i += stride) {
        v_in[i] = value;
    }

}

__global__ void
axpb_xtended(const float alpha, const float *x, const float *b, const float beta, const float *c, float *out,
             const int N, const int offset_x, const int offset_c) {
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = init; i < N; i += stride) {
        out[i] = alpha * x[i + offset_x] + b[i] + beta * c[i + offset_c];
    }
}

void Benchmark12M::alloc_vectors() {
    for (const auto &partition: this->coo_partitions) {
        f32 *tmp_vec_in, *tmp_spmv_out;
        f32 *tmp_vec_next, *tmp_lanczos_vectors, *tmp_normalized_out;
        f32 *tmp_alpha, *tmp_beta;
        CUDA_CHECK_ERROR(cudaMallocManaged(&tmp_vec_in, sizeof(f32) * this->matrix.N));
        CUDA_CHECK_ERROR(cudaMallocManaged(&tmp_spmv_out, sizeof(f32) * partition->N));
        CUDA_CHECK_ERROR(cudaMallocManaged(&tmp_vec_next, sizeof(f32) * partition->N));
        CUDA_CHECK_ERROR(
                cudaMallocManaged(&tmp_lanczos_vectors, sizeof(f32) * this->num_eigencomponents * partition->N));
        CUDA_CHECK_ERROR(cudaMallocManaged(&tmp_normalized_out, sizeof(f32) * partition->N));

        CUDA_CHECK_ERROR(cudaMallocManaged(&tmp_alpha, sizeof(f32)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&tmp_beta, sizeof(f32)));

        this->vec_in.push_back(tmp_vec_in);
        this->spmv_vec_out.push_back(tmp_spmv_out);
        this->vec_next.push_back(tmp_vec_next);
        this->lanczos_vectors.push_back(tmp_lanczos_vectors);
        this->normalized_out.push_back(tmp_normalized_out);

    }

    CUDA_CHECK_ERROR(cudaMallocManaged(&this->alpha_intermediate, sizeof(f32) * this->num_partitions));
    CUDA_CHECK_ERROR(cudaMallocManaged(&this->beta_intermediate, sizeof(f32) * this->num_partitions));

}

coo_matrix_t *Benchmark12M::assign_partition(u32 begin, u32 end, u32 last_n) {
    auto *partition = new coo_matrix_t();
    auto nnz = end - begin;
    i32 *x, *y;
    f32 *val;
    CUDA_CHECK_ERROR(cudaMallocManaged(&x, sizeof(i32) * nnz));
    CUDA_CHECK_ERROR(cudaMallocManaged(&y, sizeof(i32) * nnz));
    CUDA_CHECK_ERROR(cudaMallocManaged(&val, sizeof(f32) * nnz));

    partition->x = x;
    partition->y = y;
    partition->val = val;
    partition->nnz = nnz;
    partition->begin = begin;
    partition->end = end;

    for (u32 i = 0; i < nnz; ++i) {
        partition->x[i] = this->matrix.x[i + begin];
        partition->y[i] = this->matrix.y[i + begin] - last_n;
        partition->val[i] = this->matrix.val[i + begin];
    }

    partition->N = this->matrix.y[end - 1] - last_n;

    return partition;
}

void Benchmark12M::alloc_coo_partitions() {
    const u32 nnz_per_partition = (u32) ((f32) (this->matrix.nnz + this->num_partitions) / this->num_partitions);
    u32 begin_partition = 0;
    u32 end_partition = nnz_per_partition;
    u32 last_y_value = this->matrix.y[end_partition];
    u32 last_n_value = 0;

    for (u32 partition_id = 0; partition_id < this->num_partitions - 1; ++partition_id) {
        // Advance the partition boundary until the partition is not finished (y value changes)
        while (last_y_value == this->matrix.y[end_partition])
            end_partition++;

        // current partition is in [begin_partition, end_partition)
        auto *partition = this->assign_partition(begin_partition, end_partition, last_n_value);
        this->coo_partitions.push_back(partition);
        if (this->debug) std::cout << "Partition " << partition_id << ": " << *partition << std::endl;
        last_n_value = this->matrix.y[end_partition - 1];
        begin_partition = end_partition;
        end_partition += nnz_per_partition;
    }

    // Last partition
    auto *partition = this->assign_partition(begin_partition, this->matrix.nnz, last_n_value);
    this->coo_partitions.push_back(partition);
    if (this->debug) std::cout << "Partition " << this->num_partitions - 1 << ": " << *partition << std::endl;

}

void Benchmark12M::create_random_matrix(bool normalize = true) {
    u32 total_nnz = RANDOM_MATRIX_AVG_NNZ_PER_ROW * RANDOM_MATRIX_NUM_ROWS;
    i32 *x = (i32 *) std::malloc(total_nnz * sizeof(i32));
    i32 *y = (i32 *) std::malloc(total_nnz * sizeof(i32));
    f32 *val = (f32 *) std::malloc(total_nnz * sizeof(f32));

    for (u32 i = 0; i < total_nnz; ++i)
        val[i] = (f32) std::rand() / RAND_MAX;

    f32 acc = 0.0;
    for (u32 i = 0; i < total_nnz; ++i)
        acc += val[i] * val[i];

    f32 norm = std::sqrt(acc);

    for (u32 i = 0; i < total_nnz; ++i)
        val[i] /= norm;

    auto random_node = [&]() {
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

void Benchmark12M::load_matrix(bool normalize = true) {

    i32 M, N, nnz;
    std::vector<i32> x, y;
    std::vector<f32> val;

    i32 *x_mem, *y_mem;
    f32 *val_mem;

    read_matrix(this->matrix_path, N, M, nnz, x, y, val);
    x_mem = (i32 *) std::malloc(nnz * sizeof(i32));
    y_mem = (i32 *) std::malloc(nnz * sizeof(i32));
    val_mem = (f32 *) std::malloc(nnz * sizeof(f32));

    std::memcpy(x_mem, x.data(), nnz * sizeof(i32));
    std::memcpy(y_mem, y.data(), nnz * sizeof(i32));
    std::memcpy(val_mem, val.data(), nnz * sizeof(f32));


    this->matrix = {
            .x = x_mem,
            .y = y_mem,
            .val = val_mem,
            .begin = 0,
            .end = nnz,
            .N = N,
            .nnz = nnz
    };

    if (normalize) {

        f32 norm = std::sqrt(std::accumulate(this->matrix.val, this->matrix.val + nnz, 0.0f,
                                             [](f32 cur, f32 next) { return cur + next * next; }));
        for (u32 i = 0; i < nnz; ++i)
            this->matrix.val[i] = 1.0f / norm;

    }
}

void Benchmark12M::alloc() {

    cudaMallocManaged(&(this->alpha), sizeof(f32));
    cudaMallocManaged(&(this->beta), sizeof(f32));

    if (this->matrix_path.empty())
        this->create_random_matrix();
    else
        this->load_matrix();


    this->create_streams();
    this->alloc_coo_partitions();
    this->alloc_vectors();

    // Create offsets
    this->offsets.push_back(0);
    for (u32 i = 1; i < this->num_partitions; ++i)
        this->offsets.push_back(this->coo_partitions[i]->N + this->offsets[i - 1]);

}

void Benchmark12M::reset() {
    // Just call init, it resets all the necessary vectors;
    this->init();

    for (u32 i = 0; i < this->num_partitions; ++i) {
        const auto *partition = this->coo_partitions[i];
        for (u32 j = 0; j < partition->N; ++j) {
            this->spmv_vec_out[i][j] = 0.0f;
        }
    }
    *alpha = 0.0f;
    *beta = 0.0f;

    this->tridiagonal_matrix.clear();

}

void Benchmark12M::sync_all() {

    for (u32 i = 0; i < this->num_partitions; ++i) {
        auto selected_device = select_gpu(i, this->num_devices);
        CUDA_CHECK_ERROR(cudaSetDevice(selected_device));
        CUDA_CHECK_ERROR(cudaStreamSynchronize(this->streams[i]));
    }
    cudaSetDevice(0);
}

void Benchmark12M::create_streams() {

    for (u32 i = 0; i < this->num_partitions; ++i) {
        auto selected_device = select_gpu(i, this->num_devices);
        auto *stream = (cudaStream_t *) std::malloc(sizeof(cudaStream_t));
        CUDA_CHECK_ERROR(cudaSetDevice(selected_device));
        CUDA_CHECK_ERROR(cudaStreamCreate(stream));
        this->streams.push_back(*stream);
    }

    cudaSetDevice(0);

}

template<typename Function>
void Benchmark12M::launch_multi_kernel(Function kernel_launch_function, std::string kernel_name = "") {
    if (this->debug)
        std::cout << "Launching " << kernel_name << std::endl;
    for (u32 i = 0; i < this->num_partitions; ++i) {
        auto selected_device = select_gpu(i, this->num_devices);
        CUDA_CHECK_ERROR(cudaSetDevice(selected_device));
        const cudaStream_t &stream = streams[i];
        kernel_launch_function(i, stream);

        if (policy == Policy::Sync)
            this->sync_all();
    }

}

void Benchmark12M::execute(i32 iter) {

    if (this->debug) {
        std::cout << "[LANCZOS] Iteration " << iter << std::endl;
    }

    const auto sum = [](const f32 acc, const f32 cur) -> f32 {
        return acc + cur;
    };

    const auto sum_squares = [](const f32 acc, const f32 cur) -> f32 {
        return acc + (cur * cur);
    };

    this->launch_multi_kernel([&](u32 p_idx, const cudaStream_t &stream) {
        spmv<<<this->num_blocks, this->block_size, 0, stream>>>(
                this->coo_partitions[p_idx]->x,
                this->coo_partitions[p_idx]->y,
                this->coo_partitions[p_idx]->val,
                this->vec_in[p_idx],
                this->spmv_vec_out[p_idx],
                this->coo_partitions[p_idx]->nnz
        );
    }, "SpMV");
    //  vec_out -> 0.000878735
    //  vec_in  -> 0.000878735
    //  vec_outN-> 0.000878735



    this->launch_multi_kernel([&](u32 p_idx, const cudaStream_t &stream) {
        this->alpha_intermediate[p_idx] = 0.0;
        dot_product<<<this->num_blocks, this->block_size, 0, stream>>>(
                this->spmv_vec_out[p_idx],
                this->vec_in[p_idx],
                this->alpha_intermediate + p_idx,
                this->coo_partitions[p_idx]->N,
                this->offsets[p_idx]
        );

    }, "Dot Product");

    this->sync_all();

    *alpha = std::accumulate(this->alpha_intermediate, this->alpha_intermediate + this->num_partitions, 0.0f, sum);
    this->tridiagonal_matrix.push_back(*alpha);

    this->launch_multi_kernel([&](u32 p_idx, const cudaStream_t &stream) {
        axpb_xtended<<<this->num_blocks, this->block_size, 0, stream>>>(
                -(*alpha),
                this->vec_in[p_idx],
                this->spmv_vec_out[p_idx],
                0,
                this->vec_in[p_idx],
                this->vec_next[p_idx],
                this->coo_partitions[p_idx]->N,
                this->offsets[p_idx],
                0
        );
    }, "AXpB XTended");

    for (u32 cur_eigen = 1; cur_eigen < this->num_eigencomponents; ++cur_eigen) {

        this->launch_multi_kernel([&](u32 p_idx, const cudaStream_t &stream) {
            this->beta_intermediate[p_idx] = 0.0f;
            l2_norm_b12<<<this->num_blocks, this->block_size, 0, stream>>>(
                    this->vec_next[p_idx],
                    this->beta_intermediate + p_idx,
                    this->coo_partitions[p_idx]->N,
                    0
            );

        }, "L2 Norm");


        this->sync_all();

        *beta = std::accumulate(beta_intermediate, beta_intermediate + this->num_partitions, 0.0f, sum_squares);
        *beta = std::sqrt(*beta);

        this->tridiagonal_matrix.push_back(*beta);

        this->launch_multi_kernel([&](u32 p_idx, const cudaStream_t &stream) {
            normalize<<<this->num_blocks, this->block_size, 0, stream>>>(
                    this->vec_next[p_idx],
                    1.0f / (*beta),
                    this->normalized_out[p_idx],
                    this->coo_partitions[p_idx]->N
            );
        }, "Normalize");

        for (u32 j = 0; j < this->num_partitions; ++j) {
            this->launch_multi_kernel([&](u32 p_idx, const cudaStream_t &stream) {
                copy_partition_to_vec<<<this->num_blocks, this->block_size, 0, stream>>>(
                        this->normalized_out[p_idx],
                        this->vec_in[p_idx],
                        this->coo_partitions[p_idx]->N,
                        offsets[p_idx],
                        0
                );
            }, "copy_partition_to_vec");

            std::rotate(vec_in.begin(), vec_in.begin() + 1, vec_in.end());
        }

        if (this->num_partitions != 1)
            std::rotate(vec_in.begin(), vec_in.begin() + 1, vec_in.end());

        this->launch_multi_kernel([&](u32 p_idx, const cudaStream_t &stream) {
            set<<<this->num_blocks, this->block_size, 0, stream>>>(
                    this->spmv_vec_out[p_idx],
                    0.0f,
                    this->coo_partitions[p_idx]->N
            );
        }, "set");

        this->launch_multi_kernel([&](u32 p_idx, const cudaStream_t &stream) {
            spmv<<<this->num_blocks, this->block_size, 0, stream>>>(
                    this->coo_partitions[p_idx]->x,
                    this->coo_partitions[p_idx]->y,
                    this->coo_partitions[p_idx]->val,
                    this->vec_in[p_idx],
                    this->spmv_vec_out[p_idx],
                    this->coo_partitions[p_idx]->nnz
            );
        }, "Spmv");

        this->launch_multi_kernel([&](u32 p_idx, const cudaStream_t &stream) {
            this->alpha_intermediate[p_idx] = 0.0f;
            dot_product<<<this->num_blocks, this->block_size, 0, stream>>>(
                    this->spmv_vec_out[p_idx],
                    this->vec_in[p_idx],
                    this->alpha_intermediate + p_idx,
                    this->coo_partitions[p_idx]->N,
                    this->offsets[p_idx]
            );

        }, "Dot Product");

        this->sync_all();
        *alpha = std::accumulate(this->alpha_intermediate, this->alpha_intermediate + this->num_partitions, 0.0f, sum);
        tridiagonal_matrix.push_back(*alpha);

        this->launch_multi_kernel([&, cur_eigen](u32 p_idx, const cudaStream_t &stream) {
            axpb_xtended<<<this->num_blocks, this->block_size, 0, stream>>>(
                    -(*alpha),
                    this->vec_in[p_idx],
                    this->spmv_vec_out[p_idx],
                    -(*beta),
                    this->lanczos_vectors[p_idx],
                    this->vec_next[p_idx],
                    this->coo_partitions[p_idx]->N,
                    this->offsets[p_idx],
                    this->coo_partitions[p_idx]->N * (cur_eigen - 1)
            );
        }, "axpb_xtended");


        if (this->reorthogonalize) {

            for (u32 j = 0; j < cur_eigen; ++j) {
                this->launch_multi_kernel([&, j](u32 p_idx, const cudaStream_t &stream) {

                    dot_product<<<this->num_blocks, this->block_size, 0, stream>>>(
                            this->vec_next[p_idx],
                            this->lanczos_vectors[p_idx],
                            this->alpha,
                            this->coo_partitions[p_idx]->N,
                            this->offsets[p_idx] * j
                    );
                }, "Dot Product - Reorthogonalize");
                this->sync_all();

                this->launch_multi_kernel([&](u32 p_idx, const cudaStream_t &stream) {
                    subtract<<<this->num_blocks, this->block_size, 0, stream>>>(
                            this->vec_next[p_idx],
                            this->lanczos_vectors[p_idx],
                            *alpha,
                            this->coo_partitions[p_idx]->N,
                            this->coo_partitions[p_idx]->N
                    );
                }, "Subtract - Reorthogonalize");

            }

        }

        this->sync_all();

    }

    this->sync_all();

}

void Benchmark12M::execute_sync(i32 iter) {
    assert(this->policy == Policy::Sync);
    this->execute(iter);
}

void Benchmark12M::execute_async(int iter) {
    assert(this->policy == Policy::Async);

    for (u32 i = 0; i < this->num_partitions; ++i)
        assert(this->streams[i] != nullptr);

    this->execute(iter);
}

std::string Benchmark12M::print_result(bool short_form = false) {

    std::string base = "";
    if (this->debug) {
        for (u32 i = 0; i < this->num_eigencomponents * 2 - 1; ++i) {
            const auto &r = this->tridiagonal_matrix[i];
            base += std::to_string(i) + " -> " + std::to_string(r) + ", ";
        }

        base += "\n";
    }

    return base;
}

void Benchmark12M::init() {
    // Initialize vec_in[0]
    std::generate(this->vec_in[0], this->vec_in[0] + this->matrix.N, [this]() { return 1.0f / this->matrix.N; });

    // copy it to the other vectors
    for (u32 i = 1; i < this->num_partitions; ++i) {
        cudaMemcpy(this->vec_in[i], this->vec_in[0], this->matrix.N * sizeof(f32), cudaMemcpyHostToHost);
    }

    // Initialize the other vectors that get
    // both read and written in a single computation
    for (u32 i = 0; i < this->num_partitions; ++i) {
        const auto &partition = this->coo_partitions[i];

        for (u32 j = 0; j < partition->N; ++j) {
            this->spmv_vec_out[i][j] = 0.0f;
            this->vec_next[i][j] = 0.0f;
            this->normalized_out[i][j] = 0.0f;
        }


    }

    *alpha = 0.0f;
    *beta = 0.0f;

}

void Benchmark12M::execute_cudagraph(int iter) {
    throw new std::runtime_error("Benchmark12M::execute_cudagraph not implemented");
}

void Benchmark12M::execute_cudagraph_manual(int iter) {
    throw new std::runtime_error("Benchmark12M::execute_cudagraph_manual not implemented");
}

void Benchmark12M::execute_cudagraph_single(int iter) {
    throw new std::runtime_error("Benchmark12M::execute_cudagraph_single not implemented");
}

std::ostream &operator<<(std::ostream &os, const coo_matrix_t &matrix) {
    os << "x: " << matrix.x << " y: " << matrix.y << " val: " << matrix.val << " begin: " << matrix.begin << " end: "
       << matrix.end << " N: " << matrix.N << " nnz: " << matrix.nnz;
    return os;
}
