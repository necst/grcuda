package com.nvidia.grcuda.runtime.stream.policy;

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDALogger;
import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.oracle.truffle.api.TruffleLogger;
import org.antlr.v4.runtime.misc.NotNull;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class GrCUDAStreamPolicy {

    /**
     * Reference to the class that manages the GPU devices in this system;
     */
    protected final GrCUDADevicesManager devicesManager;
    /**
     * Total number of CUDA streams created so far;
     */
    private int totalNumberOfStreams = 0;

    private final RetrieveNewStreamPolicy retrieveNewStreamPolicy;
    private final RetrieveParentStreamPolicy retrieveParentStreamPolicy;
    protected final DeviceSelectionPolicy deviceSelectionPolicy;
    private final String bandwidthMatrixPath;

    private static final TruffleLogger STREAM_LOGGER = GrCUDALogger.getLogger(GrCUDALogger.STREAM_LOGGER);

    public GrCUDAStreamPolicy(GrCUDADevicesManager devicesManager,
                              RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum,
                              RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum,
                              DeviceSelectionPolicyEnum deviceSelectionPolicyEnum,
                              String bandwidthMatrixPath) {
        this.devicesManager = devicesManager;
        this.bandwidthMatrixPath = bandwidthMatrixPath;
        // When using a stream selection policy that supports multiple GPUs,
        // we also need a policy to choose the device where the computation is executed;
        switch (deviceSelectionPolicyEnum) {
            case ROUND_ROBIN:
                this.deviceSelectionPolicy = new RoundRobinDeviceSelectionPolicy(devicesManager);
                break;
            case STREAM_AWARE:
                this.deviceSelectionPolicy = new StreamAwareDeviceSelectionPolicy(devicesManager);
                break;
            case MIN_TRANSFER_SIZE:
                this.deviceSelectionPolicy = new MinimizeTransferSizeDeviceSelectionPolicy(devicesManager);
                break;
            case MINMIN_TRANSFER_TIME:
                this.deviceSelectionPolicy = new MinMinTransferTimeDeviceSelectionPolicy(devicesManager);
                break;
            case MINMAX_TRANSFER_TIME:
                this.deviceSelectionPolicy = new MinMaxTransferTimeDeviceSelectionPolicy(devicesManager);
                break;
            default:
                STREAM_LOGGER.finer("Disabled device selection policy, it is not necessary to use one as retrieveParentStreamPolicyEnum=" + retrieveParentStreamPolicyEnum);
                this.deviceSelectionPolicy = new SingleDeviceSelectionPolicy(devicesManager);
        }
        // Get how streams are retrieved for computations without parents;
        switch (retrieveNewStreamPolicyEnum) {
            case REUSE:
                this.retrieveNewStreamPolicy = new ReuseRetrieveStreamPolicy(this.deviceSelectionPolicy);
                break;
            case ALWAYS_NEW:
                this.retrieveNewStreamPolicy = new AlwaysNewRetrieveStreamPolicy(this.deviceSelectionPolicy);
                break;
            default:
                STREAM_LOGGER.severe("Cannot select a RetrieveNewStreamPolicy. The selected execution policy is not valid: " + retrieveNewStreamPolicyEnum);
                throw new GrCUDAException("selected RetrieveNewStreamPolicy is not valid: " + retrieveNewStreamPolicyEnum);
        }
        // Get how streams are retrieved for computations with parents;
        switch (retrieveParentStreamPolicyEnum) {
            case DISJOINT:
                this.retrieveParentStreamPolicy = new DisjointRetrieveParentStreamPolicy(this.retrieveNewStreamPolicy);
                break;
            case SAME_AS_PARENT:
                this.retrieveParentStreamPolicy = new SameAsParentRetrieveParentStreamPolicy();
                break;
            case MULTIGPU_DISJOINT:
                this.retrieveParentStreamPolicy = new MultiGPUEarlySelectionDisjointRetrieveParentStreamPolicy(this.retrieveNewStreamPolicy, this.deviceSelectionPolicy);
                break;
            default:
                STREAM_LOGGER.severe("Cannot select a RetrieveParentStreamPolicy. The selected execution policy is not valid: " + retrieveParentStreamPolicyEnum);
                throw new GrCUDAException("selected RetrieveParentStreamPolicy is not valid: " + retrieveParentStreamPolicyEnum);
        }
    }

    public GrCUDAStreamPolicy(CUDARuntime runtime,
                              RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum,
                              RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum,
                              DeviceSelectionPolicyEnum deviceSelectionPolicyEnum,
                              String bandwidthMatrixPath) {
        this(new GrCUDADevicesManager(runtime), retrieveNewStreamPolicyEnum, retrieveParentStreamPolicyEnum, deviceSelectionPolicyEnum, bandwidthMatrixPath);
    }

    /**
     * Create a new {@link CUDAStream} on the current device;
     */
    public CUDAStream createStream() {
        CUDAStream newStream = this.devicesManager.getCurrentGPU().createStream();
        this.totalNumberOfStreams++;
        return newStream;
    }

    /**
     * Create a new {@link CUDAStream} on the specified device;
     */
    public CUDAStream createStream(int gpu) {
        CUDAStream newStream = this.devicesManager.getDevice(gpu).createStream();
        this.totalNumberOfStreams++;
        return newStream;
    }

    /**
     * Obtain the stream on which to execute the input computation.
     * If the computation doesn't have any parent, obtain a new stream or a free stream.
     * If the computation has parents, we might be reuse the stream of one of the parents.
     * Each stream is uniquely associated to a single GPU. If using multiple GPUs,
     * the choice of the stream also implies the choice of the GPU where the computation is executed;
     * @param vertex the input computation for which we choose a stream;
     * @return the stream on which we execute the computation
     */
    public CUDAStream retrieveStream(ExecutionDAG.DAGVertex vertex) {
        if (vertex.isStart()) {
            // If the computation doesn't have parents, provide a new stream to it.
            // When using multiple GPUs, also select the device;
            return retrieveNewStream(vertex);
        } else {
            // Else, compute the streams used by the parent computations.
            // When using multiple GPUs, we might want to select the device as well,
            // if multiple suitable parent streams are available;
            return retrieveParentStream(vertex);
        }
    }
    
    CUDAStream retrieveNewStream(ExecutionDAG.DAGVertex vertex) {
        return this.retrieveNewStreamPolicy.retrieve(vertex);
    }

    CUDAStream retrieveParentStream(ExecutionDAG.DAGVertex vertex) {
        return this.retrieveParentStreamPolicy.retrieve(vertex);
    }

    /**
     * Update the status of a single stream within the NewStreamRetrieval policy;
     * @param stream a stream to update;
     */
    public void updateNewStreamRetrieval(CUDAStream stream) {
        this.retrieveNewStreamPolicy.update(stream);
    }

    /**
     * Update the status of all streams within the NewStreamRetrieval policy,
     * saying for example that all can be reused;
     */
    public void updateNewStreamRetrieval() {
        // All streams are free to be reused;
        this.retrieveNewStreamPolicy.update();
    }

    void cleanupNewStreamRetrieval() {
        this.retrieveNewStreamPolicy.cleanup();
    }

    /**
     * Obtain the number of streams created so far;
     */
    public int getNumberOfStreams() {
        return this.totalNumberOfStreams;
    }

    public GrCUDADevicesManager getDevicesManager() {
        return devicesManager;
    }

    /**
     * Cleanup and deallocate the streams managed by this manager;
     */
    public void cleanup() {
        this.cleanupNewStreamRetrieval();
        this.devicesManager.cleanup();
    }
    
    ///////////////////////////////////////////////////////////////
    // List of interfaces that implement RetrieveNewStreamPolicy //
    ///////////////////////////////////////////////////////////////

    /**
     * By default, create a new stream every time;
     */
    private class AlwaysNewRetrieveStreamPolicy extends RetrieveNewStreamPolicy {

        AlwaysNewRetrieveStreamPolicy(DeviceSelectionPolicy deviceSelectionPolicy) {
            super(deviceSelectionPolicy, GrCUDAStreamPolicy.this.devicesManager);
        }

        @Override
        CUDAStream retrieveStreamFromDevice(Device device) {
            return createStream(device.getDeviceId());
        }
    }

    /**
     * Keep a set of free (currently not utilized) streams, and retrieve one of them instead of always creating new streams;
     */
    private class ReuseRetrieveStreamPolicy extends RetrieveNewStreamPolicy {

        ReuseRetrieveStreamPolicy(DeviceSelectionPolicy deviceSelectionPolicy) {
            super(deviceSelectionPolicy, GrCUDAStreamPolicy.this.devicesManager);
        }

        @Override
        CUDAStream retrieveStreamFromDevice(Device device) {
            if (device.getNumberOfFreeStreams() == 0) {
                // Create a new stream if none is available;
                return createStream(device.getDeviceId());
            } else {
                return device.getFreeStream();
            }
        }
    }

    //////////////////////////////////////////////////////////////////
    // List of interfaces that implement RetrieveParentStreamPolicy //
    //////////////////////////////////////////////////////////////////

    /**
     * By default, use the same stream as the parent computation;
     */
    private static class SameAsParentRetrieveParentStreamPolicy extends RetrieveParentStreamPolicy {

        @Override
        public CUDAStream retrieve(ExecutionDAG.DAGVertex vertex) {
            return vertex.getParentComputations().get(0).getStream();
        }
    }

    /**
     * If a vertex has more than one children, each children is independent (otherwise the dependency would be added
     * from one children to the other, and not from the actual parent).
     * As such, children can be executed on different streams. In practice, this situation happens when children
     * depends on disjoint arguments subsets of the parent kernel, e.g. K1(X,Y), K2(X), K3(Y).
     * This policy re-uses the parent(s) stream(s) when possible,
     * and computes other streams using the current {@link RetrieveNewStreamPolicy};
     */
    private static class DisjointRetrieveParentStreamPolicy extends RetrieveParentStreamPolicy {
        protected final RetrieveNewStreamPolicy retrieveNewStreamPolicy;

        // Keep track of computations for which we have already re-used the stream;
        protected final Set<ExecutionDAG.DAGVertex> reusedComputations = new HashSet<>();

        public DisjointRetrieveParentStreamPolicy(RetrieveNewStreamPolicy retrieveNewStreamPolicy) {
            this.retrieveNewStreamPolicy = retrieveNewStreamPolicy;
        }

        @Override
        public CUDAStream retrieve(ExecutionDAG.DAGVertex vertex) {
            // Keep only parent vertices for which we haven't reused the stream yet;
            List<ExecutionDAG.DAGVertex> availableParents = vertex.getParentVertices().stream()
                    .filter(v -> !reusedComputations.contains(v))
                    .collect(Collectors.toList());
            // If there is at least one stream that can be re-used, take it.
            // When using multiple devices, we just take the first parent stream without considering the device of the parent;
            if (!availableParents.isEmpty()) {
                // The computation cannot be considered again;
                reusedComputations.add(availableParents.get(0));
                // Return the stream associated to this computation;
                return availableParents.get(0).getComputation().getStream();
            } else {
                // If no parent stream can be reused, provide a new stream to this computation
                //   (or possibly a free one, depending on the policy);
                return retrieveNewStreamPolicy.retrieve(vertex);
            }
        }
    }

    /**
     * This policy extends DisjointRetrieveParentStreamPolicy with multi-GPU support for reused streams,
     * not only for newly created streams.
     * In this policy, we first select the ideal GPU for the input computation.
     * Then, we find if any of the reusable streams is allocated on that device.
     * If not, we create a new stream on the ideal GPU;
     */
    private static class MultiGPUEarlySelectionDisjointRetrieveParentStreamPolicy extends DisjointRetrieveParentStreamPolicy {

        private final DeviceSelectionPolicy deviceSelectionPolicy;

        public MultiGPUEarlySelectionDisjointRetrieveParentStreamPolicy(RetrieveNewStreamPolicy retrieveNewStreamPolicy, DeviceSelectionPolicy deviceSelectionPolicy) {
            super(retrieveNewStreamPolicy);
            this.deviceSelectionPolicy = deviceSelectionPolicy;
        }

        @Override
        public CUDAStream retrieve(ExecutionDAG.DAGVertex vertex) {
            // Keep only parent vertices for which we haven't reused the stream yet;
            List<ExecutionDAG.DAGVertex> availableParents = vertex.getParentVertices().stream()
                    .filter(v -> !reusedComputations.contains(v))
                    .collect(Collectors.toList());
            // First, select the ideal device to execute this computation;
            Device selectedDevice = deviceSelectionPolicy.retrieve(vertex);

            // If at least one of the parents' streams is on the selected device, use that stream.
            // Otherwise, create a new stream on the selected device;
            if (!availableParents.isEmpty()) {
                for (ExecutionDAG.DAGVertex v : availableParents) {
                    if (v.getComputation().getStream().getStreamDeviceId() == selectedDevice.getDeviceId()) {
                        // We found a parent whose stream is on the selected device;
                        reusedComputations.add(v);
                        return v.getComputation().getStream();
                    }
                }
            }
            // If no parent stream can be reused, provide a new stream to this computation
            //   (or possibly a free one, depending on the policy);
            return retrieveNewStreamPolicy.retrieveStreamFromDevice(selectedDevice);
        }
    }

    /////////////////////////////////////////////////////////////
    // List of interfaces that implement DeviceSelectionPolicy //
    /////////////////////////////////////////////////////////////

    /**
     * With some policies (e.g. the ones that don't support multiple GPUs), we never have to perform device selection.
     * Simply return the currently active device;
     */
    public static class SingleDeviceSelectionPolicy extends DeviceSelectionPolicy {
        public SingleDeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
            super(devicesManager);
        }

        @Override
        public Device retrieve(ExecutionDAG.DAGVertex vertex) {
            return devicesManager.getCurrentGPU();
        }

        @Override
        Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            // There's only one device available, anyway;
            return this.retrieve(vertex);
        }
    }

    /**
     * Basic class for multi-GPU scheduling. Simply rotate between all the available device.
     * Not recommended for real utilization, but it can be useful for debugging
     * or as fallback for more complex policies.
     */
    public static class RoundRobinDeviceSelectionPolicy extends DeviceSelectionPolicy {
        private int nextDevice = 0;

        public RoundRobinDeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
            super(devicesManager);
        }

        private void increaseNextDevice(int startDevice) {
            this.nextDevice = (startDevice + 1) % this.devicesManager.getNumberOfGPUsToUse();
        }

        @Override
        public Device retrieve(ExecutionDAG.DAGVertex vertex) {
            Device device = this.devicesManager.getDevice(nextDevice);
            increaseNextDevice(nextDevice);
            return device;
        }

        @Override
        Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            if (devices.size() == 1) {
                Device device = devices.get(0);
                // The next device to use is the one after the only device in the list;
                increaseNextDevice(device.getDeviceId());
                return device;
            } else {
                // Sort the devices by ID;
                List<Device> sortedDevices = devices.stream().sorted(Comparator.comparingInt(Device::getDeviceId)).collect(Collectors.toList());
                // Go through the devices until we find the first device with ID bigger or equal than nextDevice;
                for (Device d : sortedDevices) {
                    if (nextDevice <= d.getDeviceId()) {
                        // The next device is the one after the current one;
                        increaseNextDevice(d.getDeviceId());
                        return d;
                    }
                }
                // If we haven't found any device, return the first device in the list
                // (i.e. reset nextDevice to 0 and start counting);
                Device device = sortedDevices.get(0);
                // Start counting from the current device;
                increaseNextDevice(device.getDeviceId());
                return device;
            }

        }
    }

    /**
     * We assign computations to the device with fewer active streams.
     * A stream is active if there's any computation assigned to it that has not been flagged as "finished".
     * The idea is to keep all devices equally busy, and avoid having devices that are used less than others;
     */
    public static class StreamAwareDeviceSelectionPolicy extends DeviceSelectionPolicy {
        public StreamAwareDeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
            super(devicesManager);
        }

        @Override
        public Device retrieve(ExecutionDAG.DAGVertex vertex) {
            return devicesManager.findDeviceWithFewerBusyStreams();
        }

        @Override
        Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            return this.retrieve(vertex);
        }
    }

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
    public static class MinimizeTransferSizeDeviceSelectionPolicy extends DeviceSelectionPolicy {

        RoundRobinDeviceSelectionPolicy roundRobin = new RoundRobinDeviceSelectionPolicy(devicesManager);

        public MinimizeTransferSizeDeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
            super(devicesManager);
        }

        @Override
        public Device retrieve(ExecutionDAG.DAGVertex vertex) {
            // Array that tracks the size, in bytes, of data that is already present on each device;
            long[] alreadyPresentDataSize = new long[devicesManager.getNumberOfGPUsToUse() + 1];
            List<AbstractArray> arguments = vertex.getComputation().getArrayArguments();
            // For each input array, compute if the array is available on other devices and does not need to be
            // transferred. We track the total size, in bytes, that is already present on each device;
            boolean isAnyDataPresentOnGPUs = false;  // True if there's at least a GPU with some data already available;
            for (AbstractArray a : arguments) {
                for (int location : a.getArrayUpToDateLocations()) {
                    if (location != CPUDevice.CPU_DEVICE_ID) {
                        alreadyPresentDataSize[location] += a.getSizeBytes();
                        isAnyDataPresentOnGPUs = true;
                    }
                }
            }
            if (isAnyDataPresentOnGPUs) {
                // Find device with maximum available data;
                int deviceWithMaximumAvailableData = 0;
                for (int i = 0; i < alreadyPresentDataSize.length; i++) {
                    deviceWithMaximumAvailableData = alreadyPresentDataSize[i] > alreadyPresentDataSize[deviceWithMaximumAvailableData] ? i : deviceWithMaximumAvailableData;
                }
                return devicesManager.getDevice(deviceWithMaximumAvailableData);
            } else {
                // FIXME: using least-busy should be better, but it is currently unreliable;
                // No data is present on any GPU: select the device with round-robin;
                return roundRobin.retrieve(vertex);
            }
        }

        @Override
        Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            return this.retrieve(vertex);
        }
    }

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
    public abstract static class TransferTimeDeviceSelectionPolicy extends DeviceSelectionPolicy {

        /**
         * Fallback policy in case no GPU has any up-tp-date data. We assume that for any GPU, transferring all the data
         * from the CPU would have the same cost, so we use this policy as tie-breaker;
         */
        private final RoundRobinDeviceSelectionPolicy roundRobin;
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

        public TransferTimeDeviceSelectionPolicy(GrCUDADevicesManager devicesManager, String bandwidthMatrixPath, java.util.function.BiFunction<Double, Double, Double> reduction, double startValue) {
            super(devicesManager);
            this.roundRobin = new RoundRobinDeviceSelectionPolicy(devicesManager);
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
                if (Integer.parseInt(records.get(il).get(0)) != -1) {
                    // GPU-GPU interconnection;
                    this.linkBandwidth[Integer.parseInt(records.get(il).get(0))][Integer.parseInt(records.get(il).get(1))] = Double.parseDouble(records.get(il).get(2));
                } else {
                    // -1 identifies CPU-GPU interconnection, store it in the last spot;
                    this.linkBandwidth[devicesManager.getNumberOfGPUsToUse()][Integer.parseInt(records.get(il).get(1))] = Double.parseDouble(records.get(il).get(2));
                    this.linkBandwidth[Integer.parseInt(records.get(il).get(1))][devicesManager.getNumberOfGPUsToUse()] = Double.parseDouble(records.get(il).get(2));
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

        @Override
        public Device retrieve(ExecutionDAG.DAGVertex vertex) {
            // Estimated transfer time if the computation is scheduled on device i-th;
            double[] argumentTransferTime = new double[devicesManager.getNumberOfGPUsToUse()];
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
            if (isAnyDataPresentOnGPUs) {
                // The best device is the one with minimum transfer time;
                int deviceWithMinimumTransferTime = 0;
                for (int i = 0; i < argumentTransferTime.length; i++) {
                    deviceWithMinimumTransferTime = argumentTransferTime[i] < argumentTransferTime[deviceWithMinimumTransferTime] ? i : deviceWithMinimumTransferTime;
                }
                return devicesManager.getDevice(deviceWithMinimumTransferTime);
            } else {
                // FIXME: using least-busy should be better, but it is currently unreliable;
                // No data is present on any GPU: select the device with round-robin;
                return roundRobin.retrieve(vertex);
            }
        }

        @Override
        Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            return this.retrieve(vertex);
        }

        public double[][] getLinkBandwidth() {
            return linkBandwidth;
        }
    }

    /**
     * Assume that data are transferred from the device that gives the best possible bandwidth.
     * In other words, find the minimum transfer time among all devices considering the minimum transfer time for each device;
     */
    public class MinMinTransferTimeDeviceSelectionPolicy extends TransferTimeDeviceSelectionPolicy {
        public MinMinTransferTimeDeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
            // Use max, we pick the maximum bandwidth between two devices;
            super(devicesManager, bandwidthMatrixPath, Math::max, 0);
        }
    }

    /**
     * Assume that data are transferred from the device that gives the worst possible bandwidth.
     * In other words, find the minimum transfer time among all devices considering the maximum transfer time for each device;
     */
    public class MinMaxTransferTimeDeviceSelectionPolicy extends TransferTimeDeviceSelectionPolicy {
        public MinMaxTransferTimeDeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
            // Use min, we pick the minimum bandwidth between two devices;
            super(devicesManager, bandwidthMatrixPath, Math::min, Double.POSITIVE_INFINITY);
        }
    }
}
