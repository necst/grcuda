#define THREADS 256
#define WIDTH 16  // shared memory width
#define HEIGHT 16 // shared memory height

#define ETA 0.3      //eta value
#define MOMENTUM 0.3 //momentum value

///////////////////////////////
///////////////////////////////

extern "C" __global__ void backprop_2(float *delta,
                                         int hid,
                                         float *ly,
                                         int in,
                                         float *w,
                                         float *oldw) {

    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    int index_y = HEIGHT * by + ty + 1;
    int index_x = tx + 1;

    w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
    oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

    __syncthreads();

    if (ty == 0 && by == 0) {
        w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
        oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    }
}