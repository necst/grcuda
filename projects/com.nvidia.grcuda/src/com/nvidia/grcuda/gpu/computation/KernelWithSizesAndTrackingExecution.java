package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.GrCUDAKernelOOBException;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.array.DeviceArray;
import com.nvidia.grcuda.gpu.ConfiguredKernel;
import com.nvidia.grcuda.gpu.KernelArgumentsWithSizesAndTracking;
import com.oracle.truffle.api.profiles.ValueProfile;

public class KernelWithSizesAndTrackingExecution extends KernelExecution {

    private final DeviceArray trackingArray;

    public KernelWithSizesAndTrackingExecution(ConfiguredKernel configuredKernel, KernelArgumentsWithSizesAndTracking args) {
        super(configuredKernel, args);
        this.trackingArray = args.getTrackingArray();
    }

    @Override
    public Object execute() {
        grCUDAExecutionContext.getCudaRuntime().cuLaunchKernel(kernel, config, args, this.getStream());
        return NoneValue.get();
    }

    @Override
    public void callback() {
        // Check if any OOB access has happened;
        for (int i = 0; i < trackingArray.getArraySize(); i++) {
            Object value = trackingArray.readArrayElementImpl(i, ValueProfile.getUncached());
            try {
                if ((int) value != 0) {
                    System.out.println("WARNING: array at position " + i + " has encountered " + value + " out-of-bounds array accesses");
                    if (grCUDAExecutionContext.getCudaRuntime().getContext().isThrowExceptionOnOOB()) {
                        throw new GrCUDAKernelOOBException("out-of-bounds array access encountered in the followin kernel execution:\n" + kernel.toString());
                    }
                }
            } catch (NullPointerException e) {
                System.out.println("WARNING: failed to check OOB presence for index " + i + ", encountered value " + value);
            }
        }
    }
}
