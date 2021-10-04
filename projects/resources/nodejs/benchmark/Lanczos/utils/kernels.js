const AXPB_XTENDED = `
extern "C" __global__ void axpb_xtended(const float alpha, const float *x, const float *b, const float beta, const float *c, float *out, const int N, const int offset_x, const int offset_c) {
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = init; i < N; i += stride) {
        out[i] = alpha * x[i + offset_x] + b[i] + beta * c[i + offset_c];
    }
}
`

const NORMALIZE = `
extern "C" __global__ void normalize(const float *d_v_in, const float denominator, float *d_v_out, int N) {
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < N; i += stride) {
        d_v_out[i] = d_v_in[i] * denominator;
    }
}
`

const STORE_AND_RESET = `
extern "C" __global__ void store_and_reset(const float *to_store, float *to_reset, float *store_destination, int N, int offset) {
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < N; i += stride) {
        store_destination[offset + i] = to_store[i];
        to_reset[i] = 0.0f;
    }
}
`

const COPY_PARTITION_TO_VEC = `
extern "C" __global__ void copy_partition_to_vec(const float *vec_in, float *vec_out, const int N, const int offset_out, const int offset_in){
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = 0; i < N; ++i){
        vec_out[i + offset_out] = vec_in[i + offset_in];
    }
}
`

const SQUARE = `
extern "C" __global__ void square(const float *x, float *y, int N){
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = init; i < N; i += stride){
        float value = x[i];
        y[i] = value * value;
    }
    
}
`

const SPMV = `
extern "C" __global__ void spmv(const int *x, const int *y, const float *val, const float *v_in, float *v_out, int num_nnz) {
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < num_nnz; i += stride) {
        v_out[y[i]] += v_in[x[i]] * val[i];
    }
}
`

const SUBTRACT = `
extern "C" __global__ void subtract(float* v1, const float* v2, const float alpha, int N, int offset) {
    int init = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = init; i < N; i += stride){
        v1[i] -= alpha * v2[i + offset];
    }    
}
`

const DOT_PRODUCT_STAGE_ONE = `
__global__ void dot_product_stage_one(const float* v1, const float* v2, float* temporaryOutputValues, int N, int offset) {
    extern __shared__ float cache[];
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
`

const DOT_PRODUCT_STAGE_TWO = `
extern "C" __global__ void dot_product_stage_two(const float *temporary_results, float *result) {
    
    float acc = temporary_results[threadIdx.x];
    for(int i = 16; i > 0; i >>= 1){
        acc += __shfl_down_sync(0xffffffff, acc, i);
        __syncthreads();
    }
    
    __syncthreads();
    
    if(threadIdx.x == 0) *result = acc;
}
`

const JACOBI_DIAGONAL = `
__device__ float atan_impl(float theta){
    static const float one_over_three = float(0.33333333333333);
    static const float one_over_five = float(0.2);
    static const float one_over_seven = float(0.142857143);

    const float t2 = theta * theta;
    const float t4 = theta * theta * theta * theta;
    const float t6 = theta * theta * theta * theta * theta * theta;

    return (1 - one_over_three * t2 + one_over_five * t4 - one_over_seven * t6) * theta;
}
__device__ float atan(float theta, float inv_theta){
    static const float pi_over_two = float(1.57079632679);
    if(theta > 1) return pi_over_two - atan_impl(inv_theta);
    if(theta < -1) return -pi_over_two - atan_impl(inv_theta);
    return atan_impl(theta);
}

__device__ float sin(float theta){
    static const float one_over_three_fact = float(0.166666667);
    static const float one_over_five_fact = float(0.008333333);

    const float t2 = theta * theta;
    const float t4 = theta * theta * theta * theta;
    return theta * (1 - t2 * one_over_three_fact + t4 * one_over_five_fact);
    }

__device__ float cos(float theta){
    static const float one_over_two_fact = float(0.5);
    static const float one_over_four_fact = float(0.041666667);
    const float t2 = theta * theta;
    const float t4 = theta * theta * theta * theta;
    return (1 - t2 * one_over_two_fact + t4 * one_over_four_fact);
}

extern "C" __global__ void jacobi_diagonal(float *matrix, float *rotation_values, const int N){
    const int core_id = 2 * blockIdx.x;
    const int core_offset = 2 * core_id;

    const int thread_position = blockDim.x + threadIdx.x;

    float theta = 0.0f;

    if(thread_position == 0) {
        // get rotation angle
        const float delta = -matrix[core_offset * (N + 1)] + matrix[(core_offset + 1) * (N + 1)];
        const float atan_arg = 2 * matrix[core_offset * (N + 1) + 1] / delta;
        const float inv_atan_arg = 0.5 * delta / matrix[core_offset * (N + 1) + 1];
        theta = atan(atan_arg, inv_atan_arg);
    }

    __syncthreads();

    const float c = cos(theta);
    const float s = sin(theta);
    const float cc = c * c;
    const float ss = s * s;

    const float two_sc = 2 * s * c;

    const float alpha = matrix[core_offset * (N + 1)];
    const float beta = matrix[core_offset * (N + 1) + 1];
    const float delta = matrix[(core_offset + 1) * (N + 1) + 1];

    switch(threadIdx.x) {
        case 0: {
            matrix[core_offset * (N + 1)] = cc * alpha - two_sc * beta + ss * delta;
            rotation_values[core_offset] = c;
            break;
        }
        case 1: {
            matrix[(core_offset + 1) * (N + 1)] = ss * alpha + two_sc * beta + cc * delta;
            rotation_values[core_offset + 1] = s;
            break;
        }
        case 2: {
            matrix[core_offset * (N + 1) + 1] = 0.0f;
            break;
        }
        case 3: {
            matrix[(core_offset + 1) * (N + 1) - 1] = 0.0f;
            break;
        }
    }




}
`


const JACOBI_OFFDIAGONAL = `
extern "C" __global__ void jacobi_offdiagonal(float *matrix, const float *rotation_values, const int N){

    const int core_idx_x = blockIdx.x;
    const int core_idx_y = blockIdx.y;
    const int core_offset_x = 2 * core_idx_x;
    const int core_offset_y = 2 * core_idx_y;
    
    const float alpha = matrix[core_offset_x * N + core_offset_y];
    const float beta  = matrix[core_offset_x * N + core_offset_y + 1];
    const float gamma = matrix[(core_offset_x + 1) * N + core_offset_y];
    const float delta = matrix[(core_offset_x + 1) * N + core_offset_y + 1];
    
    float cj = rotation_values[core_offset_x];
    float sj = rotation_values[core_offset_x + 1];
    
    float ci = rotation_values[core_offset_y];
    float si = rotation_values[core_offset_y + 1];
    
    switch(threadIdx.x * blockDim.x + threadIdx.y){
    
        case 0: {
            matrix[core_offset_x * N + core_offset_y] = cj * (alpha * ci - gamma * si) - sj * (beta * ci - delta * si);
            break;
        }
        case 1: {
            matrix[core_offset_x * N + core_offset_y + 1] = cj * (beta * ci - delta * si) + sj * (alpha * ci - gamma * si);;
            break;
        }
        case 2: {
            matrix[(core_offset_x + 1) * N + core_offset_y] = cj * (ci * gamma + alpha * si) - sj * (ci * delta + beta * si);
            break;
        }
        case 3: {
            matrix[(core_offset_x + 1) * N + core_offset_y + 1] = cj * (ci * delta + beta * si) + sj * (ci * gamma + alpha * si);
            break;
        }
    }
}

`

module.exports = {
    SPMV,
    AXPB_XTENDED,
    SQUARE,
    NORMALIZE,
    STORE_AND_RESET,
    DOT_PRODUCT_STAGE_ONE,
    DOT_PRODUCT_STAGE_TWO,
    COPY_PARTITION_TO_VEC,
    SUBTRACT,
    JACOBI_OFFDIAGONAL,
    JACOBI_DIAGONAL
}
