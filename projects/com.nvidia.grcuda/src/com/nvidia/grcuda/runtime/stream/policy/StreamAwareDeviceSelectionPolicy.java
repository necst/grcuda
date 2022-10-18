package com.nvidia.grcuda.runtime.stream.policy;

import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.Device;

import java.util.List;

/**
 * We assign computations to the device with fewer active streams.
 * A stream is active if there's any computation assigned to it that has not been flagged as "finished".
 * The idea is to keep all devices equally busy, and avoid having devices that are used less than others;
 */
public class StreamAwareDeviceSelectionPolicy extends DeviceSelectionPolicy {
    public StreamAwareDeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
        super(devicesManager);
    }

    @Override
    Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
        return devicesManager.findDeviceWithFewerBusyStreams(devices);
    }
}