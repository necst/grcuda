package com.nvidia.grcuda.runtime.stream.policy;

import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.array.AbstractArray;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Given a computation, select the device that needs the least amount of data transfer.
 * In other words, select the device that already has the maximum amount of bytes available,
 * considering the size of the input arrays.
 * For each input array, we look at the devices where the array is up to date, and give a "score"
 * to that device that is equal to the array size. Then, we pick the device with maximum score.
 * In case of ties, pick the device with lower ID.
 * We do not consider the CPU as a meaningful location, because computations cannot be scheduled on the CPU.
 * If the computation does not have any data already present on any device,
 * choose the device with round-robin selection (using {@link RoundRobinDeviceSelectionPolicy};
 */
public class MinimizeTransferSizeDeviceSelectionPolicy extends DeviceSelectionPolicy {

    /**
     * Some policies can use a threshold that specifies how much data (in percentage) must be available
     * on a device so that the device can be considered for execution.
     * A low threshold favors exploitation (using the same device for most computations),
     * while a high threshold favors exploration (distribute the computations on different devices
     * even if some device would have slightly lower synchronization time);
     */
    protected final double dataThreshold;

    /**
     * Fallback policy in case no GPU has any up-tp-date data. We assume that for any GPU, transferring all the data
     * from the CPU would have the same cost, so we use this policy as tie-breaker;
     */
    RoundRobinDeviceSelectionPolicy roundRobin = new RoundRobinDeviceSelectionPolicy(devicesManager);

    public MinimizeTransferSizeDeviceSelectionPolicy(GrCUDADevicesManager devicesManager, double dataThreshsold) {
        super(devicesManager);
        this.dataThreshold = dataThreshsold;
    }

    /**
     * For each input array of the computation, compute if the array is available on other devices and does not need to be
     * transferred. We track the total size, in bytes, that is already present on each device;
     * @param vertex the input computation
     * @param alreadyPresentDataSize the array where we store the size, in bytes, of data that is already present on each device.
     *                               The array must be zero-initialized and have size equal to the number of usable GPUs
     * @return if any data is present on any GPU. If false, we can use a fallback policy instead
     */
    boolean computeDataSizeOnDevices(ExecutionDAG.DAGVertex vertex, long[] alreadyPresentDataSize) {
        List<AbstractArray> arguments = vertex.getComputation().getArrayArguments();
        boolean isAnyDataPresentOnGPUs = false;  // True if there's at least a GPU with some data already available;
        for (AbstractArray a : arguments) {
            for (int location : a.getArrayUpToDateLocations()) {
                if (location != CPUDevice.CPU_DEVICE_ID) {
                    alreadyPresentDataSize[location] += a.getSizeBytes();
                    isAnyDataPresentOnGPUs = true;
                }
            }
        }
        return isAnyDataPresentOnGPUs;
    }

    /**
     * Find if any of the array inputs of the computation is present on the selected devices.
     * Used to understand if no device has any data already present, and the device selection policy
     * should fallback to a simpler device selection policy.
     * @param vertex the computation for which we want to select the device
     * @param devices the list of devices where the computation could be executed
     * @return if any of the computation's array inputs is already present on the specified devices
     */
    boolean isDataPresentOnGPUs(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
        for (Device d : devices) {
            for (AbstractArray a : vertex.getComputation().getArrayArguments()) {
                if (a.getArrayUpToDateLocations().contains(d.getDeviceId())) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Find if any device has at least TRANSFER_THRESHOLD % of the size of data that is required by the computation;
     * @param alreadyPresentDataSize the size in bytes that is available on each device.
     *                              The array must contain all devices in the system, not just a subset of them
     * @param vertex the computation the we analyze
     * @param devices the list of devices considered by the function
     * @return if any device has at least RANSFER_THRESHOLD % of required data
     */
    boolean findIfAnyDeviceHasEnoughData(long[] alreadyPresentDataSize, ExecutionDAG.DAGVertex vertex, List<Device> devices) {
        // Total size of the input arguments;
        long totalSize = vertex.getComputation().getArrayArguments().stream().map(AbstractArray::getSizeBytes).reduce(0L, Long::sum);
        // True if at least one device already has at least X% of the data required by the computation;
        for (Device d : devices) {
            if ((double) alreadyPresentDataSize[d.getDeviceId()] / totalSize > dataThreshold) {
                return true;
            }
        }
        return false;
    }

    /**
     * Find the device with the most bytes in it. It's just an argmax on "alreadyPresentDataSize",
     * returning the device whose ID correspond to the maximum's index
     * @param devices the list of devices to consider for the argmax
     * @param alreadyPresentDataSize the array where we store the size, in bytes, of data that is already present on each device.
     *                               the array must be zero-initialized and have size equal to the number of usable GPUs
     * @return the device with the most data
     */
    private Device selectDeviceWithMostData(List<Device> devices, long[] alreadyPresentDataSize) {
        // Find device with maximum available data;
        Device deviceWithMaximumAvailableData = devices.get(0);
        for (Device d : devices) {
            if (alreadyPresentDataSize[d.getDeviceId()] > alreadyPresentDataSize[deviceWithMaximumAvailableData.getDeviceId()]) {
                deviceWithMaximumAvailableData = d;
            }
        }
        return deviceWithMaximumAvailableData;
    }

    @Override
    public Device retrieve(ExecutionDAG.DAGVertex vertex) {
        // Array that tracks the size, in bytes, of data that is already present on each device;
        long[] alreadyPresentDataSize = new long[devicesManager.getNumberOfGPUsToUse()];
        // Compute the amount of data on each device, and if any device has any data at all;
        boolean isAnyDataPresentOnGPUs = computeDataSizeOnDevices(vertex, alreadyPresentDataSize);
        // If not device has at least X% of data available, it's not worth optimizing data locality (exploration preferred to exploitation);
        if (isAnyDataPresentOnGPUs && findIfAnyDeviceHasEnoughData(alreadyPresentDataSize, vertex, devicesManager.getUsableDevices())) {
            // Find device with maximum available data;
            return selectDeviceWithMostData(devicesManager.getUsableDevices(), alreadyPresentDataSize);
        } else {
            // No data is present on any GPU: select the device with round-robin;
            return roundRobin.retrieve(vertex);
        }
    }

    @Override
    Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
        // Array that tracks the size, in bytes, of data that is already present on each device;
        long[] alreadyPresentDataSize = new long[devicesManager.getNumberOfGPUsToUse()];
        // Compute the amount of data on each device, and if any device has any data at all;
        computeDataSizeOnDevices(vertex, alreadyPresentDataSize);
        // If not device has at least X% of data available, it's not worth optimizing data locality (exploration preferred to exploitation);
        if (findIfAnyDeviceHasEnoughData(alreadyPresentDataSize, vertex, devices)) {
            // Find device with maximum available data;
            return selectDeviceWithMostData(devices, alreadyPresentDataSize);
        } else {
            // No data is present on any GPU: select the device with round-robin;
            return roundRobin.retrieve(vertex, devices);
        }
    }
}