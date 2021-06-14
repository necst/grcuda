package com.nvidia.grcuda.gpu.computation.memAdvise;

import com.nvidia.grcuda.ComputationArgumentWithValue;
import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.stream.CUDAStream;

public class DefaultMemAdviser extends AbstractMemAdvise {

    public DefaultMemAdviser(CUDARuntime runtime) {
        super(runtime);
    }

    @Override
    public void memAdviseToGpu(GrCUDAComputationalElement computation) {
        for (ComputationArgumentWithValue a : computation.getArgumentList()) {
            if (a.getArgumentValue() instanceof AbstractArray) {
                AbstractArray array = (AbstractArray) a.getArgumentValue();
                CUDAStream streamToAdvise = computation.getStream();
                runtime.cudaMemAdvise(array, streamToAdvise.getStreamDeviceId());
            }
        }
    }
}
