package com.nvidia.grcuda.gpu.computation.prefetch;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;

public class NoneArrayPrefetcher extends AbstractArrayPrefetcher {

    public NoneArrayPrefetcher(CUDARuntime runtime) {
        super(runtime);
    }

    @Override
    public void prefetchToGpu(GrCUDAComputationalElement computation) {
    }
}
