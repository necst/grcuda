package com.nvidia.grcuda.runtime.stream.policy;

import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.Device;

import java.util.List;

/**
 * With some policies (e.g. the ones that don't support multiple GPUs), we never have to perform device selection.
 * Simply return the currently active device;
 */
public class SingleDeviceSelectionPolicy extends DeviceSelectionPolicy {
    public SingleDeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
        super(devicesManager);
    }

    @Override
    Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
        // There's only one device available, anyway;
        return devicesManager.getCurrentGPU();
    }
}