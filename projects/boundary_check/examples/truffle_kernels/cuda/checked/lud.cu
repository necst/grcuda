#define BLOCK_SIZE 16

// OoB accesses  are present, they must be avoided by setting the right values of offset and number of blocks.
// The dimension of input matrix should be multiple size of block size;
extern "C" __global__ void lud_perimeter(float *m, int matrix_dim, int offset) {
    __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

    int i, j, array_offset;
    int idx;

    if (threadIdx.x < BLOCK_SIZE) {
        idx = threadIdx.x;

        array_offset = offset * matrix_dim + offset;
        for (i = 0; i < BLOCK_SIZE / 2; i++) {
            dia[i][idx] = m[array_offset + idx];
            array_offset += matrix_dim;
        }

        array_offset = offset * matrix_dim + offset;
        for (i = 0; i < BLOCK_SIZE; i++) {
            peri_row[i][idx] = m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx];
            array_offset += matrix_dim;
        }

    } else {
        idx = threadIdx.x - BLOCK_SIZE;

        array_offset = (offset + BLOCK_SIZE / 2) * matrix_dim + offset;
        for (i = BLOCK_SIZE / 2; i < BLOCK_SIZE; i++) {
            dia[i][idx] = m[array_offset + idx];
            array_offset += matrix_dim;
        }

        array_offset = (offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + offset;
        for (i = 0; i < BLOCK_SIZE; i++) {
            peri_col[i][idx] = m[array_offset + idx];
            array_offset += matrix_dim;
        }
    }
    __syncthreads();

    if (threadIdx.x < BLOCK_SIZE) { //peri-row
        idx = threadIdx.x;
        for (i = 1; i < BLOCK_SIZE; i++) {
            for (j = 0; j < i; j++)
                peri_row[i][idx] -= dia[i][j] * peri_row[j][idx];
        }
    } else { //peri-col
        idx = threadIdx.x - BLOCK_SIZE;
        for (i = 0; i < BLOCK_SIZE; i++) {
            for (j = 0; j < i; j++)
                peri_col[idx][i] -= peri_col[idx][j] * dia[j][i];
            peri_col[idx][i] /= dia[i][i];
        }
    }

    __syncthreads();

    if (threadIdx.x < BLOCK_SIZE) { //peri-row
        idx = threadIdx.x;
        array_offset = (offset + 1) * matrix_dim + offset;
        for (i = 1; i < BLOCK_SIZE; i++) {
            m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx] = peri_row[i][idx];
            array_offset += matrix_dim;
        }
    } else { //peri-col
        idx = threadIdx.x - BLOCK_SIZE;
        array_offset = (offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + offset;
        for (i = 0; i < BLOCK_SIZE; i++) {
            m[array_offset + idx] = peri_col[i][idx];
            array_offset += matrix_dim;
        }
    }
}