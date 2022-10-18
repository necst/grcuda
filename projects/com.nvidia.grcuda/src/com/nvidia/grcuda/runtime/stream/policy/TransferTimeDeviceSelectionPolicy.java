package com.nvidia.grcuda.runtime.stream.policy;

import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.array.AbstractArray;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileReader;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;


/**
 * Given a computation, select the device that requires the least time to transfer data to it.
 * Compared to {@link MinimizeTransferSizeDeviceSelectionPolicy} this policy does not simply select the
 * device that requires the least data to be transferred to it, but also estimate the time that it takes
 * to transfer the data, given a heterogeneous multi-GPU system.
 * Given the complexity of CUDA's unified memory heuristics, we allow different heuristics to be used to estimate
 * the actual transfer speed, e.g. take the min or max possible values;
 * The speed of each GPU-GPU and CPU-GPU link is assumed to be stored in a file located in "$GRCUDA_HOME/grcuda_data/datasets/connection_graph/connection_graph.csv".
 * This file is generated as "cd $GRCUDA_HOME/projects/resources/cuda", "make connection_graph", "bin/connection_graph";
 */
public abstract class TransferTimeDeviceSelectionPolicy extends MinimizeTransferSizeDeviceSelectionPolicy {

    /**
     * This functional tells how the transfer bandwidth for some array and device is computed.
     * It should be max, min, mean, etc.;
     */
    private final java.util.function.BiFunction<Double, Double, Double> reduction;
    /**
     * Starting value of the reduction. E.g. it can be 0 if using max or mean, +inf if using min, etc.
     */
    private final double startValue;

    private final double[][] linkBandwidth = new double[devicesManager.getNumberOfGPUsToUse() + 1][devicesManager.getNumberOfGPUsToUse() + 1];

    public TransferTimeDeviceSelectionPolicy(GrCUDADevicesManager devicesManager, double dataThreshold, String bandwidthMatrixPath, java.util.function.BiFunction<Double, Double, Double> reduction, double startValue) {
        super(devicesManager, dataThreshold);
        this.reduction = reduction;
        this.startValue = startValue;

        List<List<String>> records = new ArrayList<>();
        // Read each line in the CSV and store each line into a string array, splitting strings on ",";
        try (BufferedReader br = new BufferedReader(new FileReader(bandwidthMatrixPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                records.add(Arrays.asList(values));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        // Read each line, and reconstruct the bandwidth matrix.
        // Given N GPUs and 1 CPU, we have a [GPU + 1][GPU+ 1] symmetric matrix.
        // Each line is "start_id", "end_id", "bandwidth";
        for (int il = 1; il < records.size(); il++) {
            int startDevice = Integer.parseInt(records.get(il).get(0));
            int endDevice = Integer.parseInt(records.get(il).get(1));
            // Skip invalid entries, and ignore GPUs with ID larger than the number of GPUs to use;
            if (startDevice >= -1 && startDevice < devicesManager.getNumberOfGPUsToUse()
                    && endDevice >= -1 && endDevice < devicesManager.getNumberOfGPUsToUse()) {
                // Approximate to the floor, to smooth random bandwidth fluctuations in data transfer;
                double bandwidth = Math.floor(Double.parseDouble(records.get(il).get(2)));
                if (startDevice != -1) {
                    // GPU-GPU interconnection;
                    this.linkBandwidth[startDevice][endDevice] = bandwidth;
                } else {
                    // -1 identifies CPU-GPU interconnection, store it in the last spot;
                    this.linkBandwidth[devicesManager.getNumberOfGPUsToUse()][endDevice] = bandwidth;
                    this.linkBandwidth[endDevice][devicesManager.getNumberOfGPUsToUse()] = bandwidth;
                }
            }
        }
        // Interconnections are supposedly symmetric. Enforce this behavior by averaging results.
        // In other words, B[i][j] = B[j][j] <- (B[i][j] + B[j][i]) / 2.
        // Ignore the diagonal, and the last column and row (it represents the CPU and is already symmetric by construction);
        for (int i = 0; i < this.linkBandwidth.length - 1; i++) {
            for (int j = i; j < this.linkBandwidth.length - 1; j++) {
                double averageBandwidth = (this.linkBandwidth[i][j] + this.linkBandwidth[j][i]) / 2;
                this.linkBandwidth[i][j] = averageBandwidth;
                this.linkBandwidth[j][i] = averageBandwidth;
            }
        }
    }

    /**
     * Estimate the bandwidth to transfer data to a "targetDevice" GPU, assuming
     * that data can be transferred from devices with index in "upToDateLocations".
     * @param targetDevice where we want to transfer data
     * @param upToDateLocations from where we can transfer data
     * @return an estimate of the transfer bandwidth
     */
    public double computeBandwidth(int targetDevice, Set<Integer> upToDateLocations) {
        // Hypotheses: we consider the max bandwidth towards the device targetDevice.
        // Initialization: min possible value, bandwidth = 0 GB/sec;
        double bandwidth = startValue;
        // Check that data is updated at least in some location. This is a precondition that must hold;
        if (upToDateLocations == null || upToDateLocations.isEmpty()) {
            throw new IllegalStateException("data is not updated in any location, when estimating bandwidth for device=" + targetDevice);
        }
        // If array a already present in device targetDevice, the transfer bandwidth to it is infinity.
        // We don't need to transfer it, so its transfer time will be 0;
        if (upToDateLocations.contains(targetDevice)) {
            bandwidth = Double.POSITIVE_INFINITY;
        } else {
            // Otherwise we consider the bandwidth to move array a to device targetDevice,
            // from each possible location containing the array a;
            for (int location : upToDateLocations) {
                // The CPU bandwidth is stored in the last column;
                int fromDevice = location != CPUDevice.CPU_DEVICE_ID ? location : linkBandwidth.length - 1;
                // The matrix is symmetric, loading [targetDevice][fromDevice] is faster than [fromDevice][targetDevice];
                bandwidth = reduction.apply(linkBandwidth[targetDevice][fromDevice], bandwidth);
            }
        }
        return bandwidth;
    }

    /**
     * For each device, measure how long it takes to transfer the data that is required
     * to run the computation in vertex
     * @param vertex the computation that we want to run
     * @param argumentTransferTime the array where we store the time, in seconds, to transfer the required data on each device
     *                             The array must be zero-initialized and have size equal to the number of usable GPUs
     * @return if any data is present on any GPU. If false, we can use a fallback policy instead
     */
    private boolean computeTransferTimes(ExecutionDAG.DAGVertex vertex, double[] argumentTransferTime) {
        List<AbstractArray> arguments = vertex.getComputation().getArrayArguments();

        // True if there's at least a GPU with some data already available;
        boolean isAnyDataPresentOnGPUs = false;

        // For each input array, consider how much time it takes to transfer it from every other device;
        for (AbstractArray a : arguments) {
            Set<Integer> upToDateLocations = a.getArrayUpToDateLocations();
            if (upToDateLocations.size() > 1 || (upToDateLocations.size() == 1 && !upToDateLocations.contains(CPUDevice.CPU_DEVICE_ID))) {
                isAnyDataPresentOnGPUs = true;
            }
            // Check all available GPUs and compute the tentative transfer time for each of them.
            // to find the device where transfer time is minimum;
            for (int i = 0; i < argumentTransferTime.length; i++) {
                // Add estimated transfer time;
                argumentTransferTime[i] += a.getSizeBytes() / computeBandwidth(i, upToDateLocations);
            }
        }
        return isAnyDataPresentOnGPUs;
    }

    /**
     * Find the devices with at least TRANSFER_THRESHOLD % of the size of data that is required by the computation;
     * @param alreadyPresentDataSize the size in bytes that is available on each device.
     *                              The array must contain all devices in the system, not just a subset of them
     * @param vertex the computation the we analyze
     * @param devices the list of devices considered by the function
     * @return the list of devices that has at least TRANSFER_THRESHOLD % of required data
     */
    List<Device> findDevicesWithEnoughData(long[] alreadyPresentDataSize, ExecutionDAG.DAGVertex vertex, List<Device> devices) {
        // List of devices with enough data;
        List<Device> devicesWithEnoughData = new ArrayList<>();
        // Total size of the input arguments;
        long totalSize = vertex.getComputation().getArrayArguments().stream().map(AbstractArray::getSizeBytes).reduce(0L, Long::sum);
        // True if at least one device already has at least X% of the data required by the computation;
        for (Device d : devices) {
            if ((double) alreadyPresentDataSize[d.getDeviceId()] / totalSize > dataThreshold) {
                devicesWithEnoughData.add(d);
            }
        }
        return devicesWithEnoughData;
    }

    /**
     * Find the device with the lower synchronization time. It's just an argmin on "argumentTransferTime",
     * returning the device whose ID correspond to the minimum's index
     * @param devices the list of devices to consider for the argmin
     * @param argumentTransferTime the array where we store the time, in seconds, to transfer the required data on each device
     *                             The array must be zero-initialized and have size equal to the number of usable GPUs
     * @return the device with the most data
     */
    private Device findDeviceWithLowestTransferTime(List<Device> devices, double[] argumentTransferTime) {
        // The best device is the one with minimum transfer time;
        Device deviceWithMinimumTransferTime = devices.get(0);
        for (Device d : devices) {
            if (argumentTransferTime[d.getDeviceId()] < argumentTransferTime[deviceWithMinimumTransferTime.getDeviceId()]) {
                deviceWithMinimumTransferTime = d;
            }
        }
        return deviceWithMinimumTransferTime;
    }

    @Override
    public Device retrieve(ExecutionDAG.DAGVertex vertex) {
        return this.retrieveImpl(vertex, devicesManager.getUsableDevices());
    }

    @Override
    Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
        // Estimated transfer time if the computation is scheduled on device i-th;
        double[] argumentTransferTime = new double[devicesManager.getNumberOfGPUsToUse()];
        // Compute the synchronization time on each device, and if any device has any data at all;
        boolean isAnyDataPresentOnGPUs = computeTransferTimes(vertex, argumentTransferTime);
        List<Device> devicesWithEnoughData = new ArrayList<>();
        if (isAnyDataPresentOnGPUs) {  // Skip this step if no GPU has any data in it;
            // Array that tracks the size, in bytes, of data that is already present on each device;
            long[] alreadyPresentDataSize = new long[devicesManager.getNumberOfGPUsToUse()];
            // Compute the amount of data on each device, and if any device has any data at all;
            computeDataSizeOnDevices(vertex, alreadyPresentDataSize);
            // Compute the list of devices that have at least X% of data already available;
            devicesWithEnoughData = findDevicesWithEnoughData(alreadyPresentDataSize, vertex, devices);
        }
        // If no device has at least X% of data available, it's not worth optimizing data locality (exploration preferred to exploitation);
        if (isAnyDataPresentOnGPUs && !devicesWithEnoughData.isEmpty()) {
            // The best device is the one with minimum transfer time;
            return findDeviceWithLowestTransferTime(devicesWithEnoughData, argumentTransferTime);
        } else {
            // No data is present on any GPU: select the device with round-robin;
            return roundRobin.retrieve(vertex, devices);
        }
    }

    public double[][] getLinkBandwidth() {
        return linkBandwidth;
    }

/**
 * Assume that data are transferred from the device that gives the best possible bandwidth.
 * In other words, find the minimum transfer time among all devices considering the minimum transfer time for each device;
 */
public static class MinMinTransferTimeDeviceSelectionPolicy extends TransferTimeDeviceSelectionPolicy {
    public MinMinTransferTimeDeviceSelectionPolicy(GrCUDADevicesManager devicesManager, double dataThreshold, String bandwidthMatrixPath) {
        // Use max, we pick the maximum bandwidth between two devices;
        super(devicesManager, dataThreshold, bandwidthMatrixPath, Math::max, 0);
    }
}

/**
 * Assume that data are transferred from the device that gives the worst possible bandwidth.
 * In other words, find the minimum transfer time among all devices considering the maximum transfer time for each device;
 */
public static class MinMaxTransferTimeDeviceSelectionPolicy extends TransferTimeDeviceSelectionPolicy {
    public MinMaxTransferTimeDeviceSelectionPolicy(GrCUDADevicesManager devicesManager, double dataThreshold, String bandwidthMatrixPath) {
        // Use min, we pick the minimum bandwidth between two devices;
        super(devicesManager, dataThreshold, bandwidthMatrixPath, Math::min, Double.POSITIVE_INFINITY);
    }
}
}