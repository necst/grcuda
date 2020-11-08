extern "C" __global__ void huffman(int *srcData, int *cindex, int *cindex2, int *dstData, int original_num_block_elements) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // source index
    int offset = tid * original_num_block_elements; //DPB,
    int bitsize = cindex[tid];

    // destination index
    int pos = cindex2[tid],
        dword = pos / 32,
        bit = pos % 32;

    int i, dw, tmp;
    dw = srcData[offset];                // load the first dword from srcData[]
    tmp = dw >> bit;                     // cut off those bits that do not fit into the initial location in destData[]
    atomicOr(&dstData[dword], tmp);      // fill up this initial location
    tmp = dw << (32 - bit);              // save the remaining bits that were cut off earlier in tmp
    for (i = 1; i < bitsize / 32; i++) { // from now on, we have exclusive access to destData[]
        dw = srcData[offset + i];        // load next dword from srcData[]
        tmp |= dw >> bit;                // fill up tmp
        dstData[dword + i] = tmp;        // write complete dword to destData[]
        tmp = dw << (32 - bit);          // save the remaining bits in tmp (like before)
    }
    // exclusive access to dstData[] ends here
    // the remaining block can, or rather should be further optimized
    // write the remaining bits in tmp, UNLESS bit is 0 and bitsize is divisible by 32, in this case do nothing
    if (bit != 0 || bitsize % 32 != 0)
        atomicOr(&dstData[dword + i], tmp);
    if (bitsize % 32 != 0) {
        dw = srcData[offset + i];
        atomicOr(&dstData[dword + i], dw >> bit);
        atomicOr(&dstData[dword + i + 1], dw << (32 - bit));
    }
}

extern "C" __global__ void huffman_checked(int *srcData, int *cindex, int *cindex2, int *dstData, int original_num_block_elements, int index_size, int num_elements) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // source index
    int offset = tid * original_num_block_elements; //DPB,
    if (tid < index_size) {
        int bitsize = cindex[tid];

        // destination index
        int pos = cindex2[tid],
            dword = pos / 32,
            bit = pos % 32;

        int i, dw, tmp;
        if (offset < num_elements && dword < num_elements) {
            dw = srcData[offset];           // load the first dword from srcData[]
            tmp = dw >> bit;                // cut off those bits that do not fit into the initial location in destData[]
            atomicOr(&dstData[dword], tmp); // fill up this initial location
            tmp = dw << (32 - bit);
            if (dword + bitsize / 32 + 1 < num_elements && offset + bitsize / 32 < num_elements) {
                for (i = 1; i < bitsize / 32; i++) {
                    dw = srcData[offset + i];
                    tmp |= dw >> bit;
                    dstData[dword + i] = tmp;
                    tmp = dw << (32 - bit);
                }
                // exclusive access to dstData[] ends here
                // the remaining block can, or rather should be further optimized
                // write the remaining bits in tmp, UNLESS bit is 0 and bitsize is divisible by 32, in this case do nothing
                if (bit != 0 || bitsize % 32 != 0)
                    atomicOr(&dstData[dword + i], tmp);
                if (bitsize % 32 != 0) {
                    dw = srcData[offset + i];
                    atomicOr(&dstData[dword + i], dw >> bit);
                    atomicOr(&dstData[dword + i + 1], dw << (32 - bit));
                }
            }
        }
    }
}

__global__ void huffman2(unsigned int *data,
                         const unsigned int *gm_codewords, const unsigned int *gm_codewordlens,
                         unsigned int *out, unsigned int *outidx) {

    unsigned int kn = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = threadIdx.x;
    unsigned int kc, startbit, wrbits;

    unsigned long long cw64 = 0;
    unsigned int val32, codewordlen = 0;
    unsigned char tmpbyte, tmpcwlen;
    unsigned int tmpcw32;

    extern __shared__ unsigned int sm[];
    __shared__ unsigned int kcmax;

    unsigned int *as = (unsigned int *)sm;
    val32 = data[kn];
    for (unsigned int i = 0; i < 4; i++) {
        tmpbyte = (unsigned char)(val32 >> ((3 - i) * 8));
        tmpcw32 = gm_codewords[tmpbyte];
        tmpcwlen = gm_codewordlens[tmpbyte];
        cw64 = (cw64 << tmpcwlen) | tmpcw32;
        codewordlen += tmpcwlen;
    }

    as[k] = codewordlen;
    __syncthreads();

    /* Prefix sum of codeword lengths (denoted in bits) [inplace implementation] */
    unsigned int offset = 1;

    /* Build the sum in place up the tree */
    for (unsigned int d = (blockDim.x) >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (k < d) {
            unsigned char ai = offset * (2 * k + 1) - 1;
            unsigned char bi = offset * (2 * k + 2) - 1;
            as[bi] += as[ai];
        }
        offset *= 2;
    }

    /* scan back down the tree */
    /* clear the last element */
    if (k == 0)
        as[blockDim.x - 1] = 0;

    // traverse down the tree building the scan in place
    for (unsigned int d = 1; d < blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (k < d) {
            unsigned char ai = offset * (2 * k + 1) - 1;
            unsigned char bi = offset * (2 * k + 2) - 1;
            unsigned int t = as[ai];
            as[ai] = as[bi];
            as[bi] += t;
        }
    }
    __syncthreads();

    if (k == blockDim.x - 1) {
        outidx[blockIdx.x] = as[k] + codewordlen;
        kcmax = (as[k] + codewordlen) / 32;
    }

    /* Write the codes */
    kc = as[k] / 32;
    startbit = as[k] % 32;
    as[k] = 0U;
    __syncthreads();

    /* Part 1*/
    wrbits = codewordlen > (32 - startbit) ? (32 - startbit) : codewordlen;
    tmpcw32 = (unsigned int)(cw64 >> (codewordlen - wrbits));
    //if (wrbits == 32) as[kc] = tmpcw32;				//unnecessary overhead; increases number of branches
    //else
    atomicOr(&as[kc], tmpcw32 << (32 - startbit - wrbits)); //shift left in case it's shorter then the available space
    codewordlen -= wrbits;

    /*Part 2*/
    if (codewordlen) {
        wrbits = codewordlen > 32 ? 32 : codewordlen;
        tmpcw32 = (unsigned int)(cw64 >> (codewordlen - wrbits)) & ((1 << wrbits) - 1);
        //if (wrbits == 32) as[kc+1] = tmpcw32;
        //else
        atomicOr(&as[kc + 1], tmpcw32 << (32 - wrbits));
        codewordlen -= wrbits;
    }

    /*Part 3*/
    if (codewordlen) {
        tmpcw32 = (unsigned int)(cw64 & ((1 << codewordlen) - 1));
        //if (wrbits == 32) as[kc+2] = tmpcw32;
        //else
        atomicOr(&as[kc + 2], tmpcw32 << (32 - codewordlen));
    }

    __syncthreads();

    if (k <= kcmax)
        out[kn] = as[k];
}
