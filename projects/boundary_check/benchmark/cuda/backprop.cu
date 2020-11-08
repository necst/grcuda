#define THREADS 256
#define WIDTH 16  // shared memory width
#define HEIGHT 16 // shared memory height

///////////////////////////////
///////////////////////////////

extern "C" __global__ void
backprop(float *input_cuda,
                       float *output_hidden_cuda,
                       float *input_hidden_cuda,
                       float *hidden_partial_sum,
                       int in,
                       int hid) {

    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);

    int index_in = HEIGHT * by + ty + 1;

    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    // Unsafe access;
    if (tx == 0) {
        input_node[ty] = input_cuda[index_in];
    }

    __syncthreads();

    // Unsafe access;
    weight_matrix[ty][tx] = input_hidden_cuda[index];

    __syncthreads();

    weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

    __syncthreads();

    for (int i = 1; i <= __log2f(HEIGHT); i++) {

        int power_two = __powf(2, i);

        if (ty % power_two == 0) {
            weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two / 2][tx];
        }

        __syncthreads();
    }

    input_hidden_cuda[index] = weight_matrix[ty][tx];

    __syncthreads();

    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
    }
}


