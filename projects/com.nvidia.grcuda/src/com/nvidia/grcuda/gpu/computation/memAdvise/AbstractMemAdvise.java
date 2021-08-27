package com.nvidia.grcuda.gpu.computation.memAdvise;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;

/**
 * Class that declares an interface to use memAdvise function to move memory region between CPU and GPU.
 */
public abstract class AbstractMemAdvise {

    protected CUDARuntime runtime;

    public AbstractMemAdvise(CUDARuntime runtime) {
        this.runtime = runtime;
    }

    /**
     * advise the arrays of a {@link GrCUDAComputationalElement}.
     * @param computation a computational element whose array inputs can be moved from host to GPU
     */
    public abstract void memAdviseToGpu(GrCUDAComputationalElement computation);

    public void memAdviseToGpu(ExecutionDAG.DAGVertex vertex) {
        this.memAdviseToGpu(vertex.getComputation());
    }
}
