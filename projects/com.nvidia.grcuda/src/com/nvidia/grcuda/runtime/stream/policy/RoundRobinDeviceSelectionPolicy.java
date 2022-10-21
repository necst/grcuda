package com.nvidia.grcuda.runtime.stream.policy;

import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.Device;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Basic class for multi-GPU scheduling. Simply rotate between all the available device.
 * Not recommended for real utilization, but it can be useful for debugging
 * or as fallback for more complex policies.
 */
public class RoundRobinDeviceSelectionPolicy extends DeviceSelectionPolicy {
    private int nextDevice = 0;

    public RoundRobinDeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
        super(devicesManager);
    }

    private void increaseNextDevice(int startDevice) {
        this.nextDevice = (startDevice + 1) % this.devicesManager.getNumberOfGPUsToUse();
    }

    public int getInternalState() {
        return nextDevice;
    }

    @Override
    Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
        // Keep increasing the internal state, but make sure that the retrieved device is among the ones in the input list;
        Device device = devices.get(nextDevice % devices.size());
        increaseNextDevice(nextDevice);
        return device;
    }
}