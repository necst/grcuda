package com.nvidia.grcuda.gpu.computation.memAdvise;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;

public class NoneMemAdviser extends AbstractMemAdvise {

    public NoneMemAdviser(CUDARuntime runtime) {
        super(runtime);
    }

    /**
     * This array prefetcher doesn't do anything;
     * @param computation a computational element whose array inputs can be prefetched from host to GPU
     */
    @Override
    public void memAdviseToGpu(GrCUDAComputationalElement computation) {
    }
}
