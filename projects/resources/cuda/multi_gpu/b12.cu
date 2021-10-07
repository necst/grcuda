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

void Benchmark12::alloc_vectors() {
    for (const auto& partition: this->coo_partitions){
        f32 *tmp_vec_in, *tmp_spmv_out, *tmp_intermediate_dot_product_values;
        f32 *tmp_alpha_intermediate, *tmp_beta_intermediate;
        f32 *tmp_vec_next, *tmp_lanczos_vectors, *tmp_normalized_out;

        cudaMallocManaged(&tmp_vec_in, sizeof(f32) * this->matrix.N);
        cudaMallocManaged(&tmp_spmv_out, sizeof(f32) * partition.N);
        cudaMallocManaged(&tmp_intermediate_dot_product_values, sizeof(f32) * 32);
        cudaMallocManaged(&tmp_alpha_intermediate, sizeof(f32) * this->num_gpus);
        cudaMallocManaged(&tmp_beta_intermediate, sizeof(f32) * this->num_gpus);
        cudaMallocManaged(&tmp_vec_next, sizeof(f32) * partition.N);
        cudaMallocManaged(&tmp_lanczos_vectors, sizeof(f32) * this->num_eigencomponents * partition.N);
        cudaMallocManaged(&tmp_normalized_out, sizeof(f32) * partition.N);

        this->vec_in.push_back(tmp_vec_in);
        this->spmv_vec_out.push_back(tmp_spmv_out);
        this->intermediate_dot_product_values.push_back(tmp_intermediate_dot_product_values);
        this->alpha_intermediate.push_back(tmp_alpha_intermediate);
        this->beta_intermediate.push_back(tmp_beta_intermediate);
        this->vec_next.push_back(tmp_vec_next);
        this->lanczos_vectors.push_back(tmp_lanczos_vectors);
        this->normalized_out.push_back(tmp_normalized_out);
    }
}

void Benchmark12::alloc_coo_partitions() {

    const u32 nnz_per_partition = u32((this->matrix.nnz + this->num_gpus) / this->num_gpus);
    u32 from_index = 0;
    u32 to_index = nnz_per_partition;
    u32 index_value = this->matrix.y[to_index];

    for(u32 i = 0; i < this->num_gpus; ++i){
        while(index_value == this->matrix.y[to_index]) {
            to_index++;
        }
        const u32 offset = (from_index == 0) ? from_index : (this->matrix.y[from_index] - 1);
        auto coo_partition = this->assign_partition(from_index, to_index, offset);
        this->coo_partitions.push_back(coo_partition);

        from_index = to_index;
        to_index += nnz_per_partition;
        index_value = this->matrix.y[to_index];
    }

    const u32 offset = this->matrix.y[from_index];
    auto coo_partition = this->assign_partition(from_index, this->matrix.nnz, offset);
    this->coo_partitions.push_back(coo_partition);
}

coo_matrix_t Benchmark12::assign_partition(u32 from_index, u32 to_index, u32 offset) {
    coo_matrix_t coo_partition;
    coo_partition.begin = from_index;
    coo_partition.end = to_index;
    cudaMallocManaged(&coo_partition.x, sizeof(u32) * (to_index - from_index));
    cudaMallocManaged(&coo_partition.y, sizeof(u32) * (to_index - from_index));
    cudaMallocManaged(&coo_partition.val, sizeof(u32) * (to_index - from_index));

    for(u32 i = from_index; i < to_index; ++i){
        coo_partition.x[i]   = this->matrix.x[i];
        coo_partition.y[i]   = this->matrix.y[i];
        coo_partition.val[i] = this->matrix.val[i];
    }

    coo_partition.N = coo_partition.y[to_index - from_index - 1];

    return coo_partition;
}


void Benchmark12::alloc() {

    this->alloc_coo_partitions();
    this->alloc_vectors();

}

void Benchmark12::reset() {}
void Benchmark12::execute_sync(int iter) {
    // TODO: implement
}
void Benchmark12::execute_async(int iter) {
    // TODO: implement
}

std::string Benchmark12::print_result(bool short_form = false) {
    return "";
}

void Benchmark12::init() {
    // TODO: initialize the vectors with appropriate values (either zero or random)
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