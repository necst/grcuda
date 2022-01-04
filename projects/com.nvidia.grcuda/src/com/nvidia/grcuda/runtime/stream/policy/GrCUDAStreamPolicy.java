package com.nvidia.grcuda.runtime.stream.policy;

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDALogger;
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.oracle.truffle.api.TruffleLogger;

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

    private static final TruffleLogger STREAM_LOGGER = GrCUDALogger.getLogger(GrCUDALogger.STREAM_LOGGER);

    public GrCUDAStreamPolicy(GrCUDADevicesManager devicesManager,
                              RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum,
                              RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum,
                              DeviceSelectionPolicyEnum deviceSelectionPolicyEnum) {
        this.devicesManager = devicesManager;
        // When using a stream selection policy that supports multiple GPUs,
        // we also need a policy to choose the device where the computation is executed;
        DeviceSelectionPolicy deviceSelectionPolicy;
        switch (deviceSelectionPolicyEnum) {
            case STREAM_AWARE:
                deviceSelectionPolicy = new StreamAwareDeviceSelectionPolicy();
                break;
            default:
                STREAM_LOGGER.finer("Disabled device selection policy, it is not necessary to use one as retrieveParentStreamPolicyEnum=" + retrieveParentStreamPolicyEnum);
                deviceSelectionPolicy = new SingleDeviceSelectionPolicy();
        }
        // Get how streams are retrieved for computations without parents;
        switch (retrieveNewStreamPolicyEnum) {
            case REUSE:
                this.retrieveNewStreamPolicy = new ReuseRetrieveStreamPolicy(deviceSelectionPolicy);
                break;
            case ALWAYS_NEW:
                this.retrieveNewStreamPolicy = new AlwaysNewRetrieveStreamPolicy(deviceSelectionPolicy);
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
                this.retrieveParentStreamPolicy = new MultiGPUEarlySelectionDisjointRetrieveParentStreamPolicy(this.retrieveNewStreamPolicy, deviceSelectionPolicy);
                break;
            default:
                STREAM_LOGGER.severe("Cannot select a RetrieveParentStreamPolicy. The selected execution policy is not valid: " + retrieveParentStreamPolicyEnum);
                throw new GrCUDAException("selected RetrieveParentStreamPolicy is not valid: " + retrieveParentStreamPolicyEnum);
        }

//        if (retrieveParentStreamPolicyEnum == RetrieveParentStreamPolicyEnum.MULTIGPU_DATA_AWARE ||
//                retrieveParentStreamPolicyEnum == RetrieveParentStreamPolicyEnum.MULTIGPU_DISJOINT_DATA_AWARE) {
//            switch (deviceSelectionPolicyEnum) {
//                case DATA_LOCALITY_NEW:
//                    this.deviceSelectionPolicy = new MoveFewerArgumentsNewDeviceSelectionPolicy();
//                    break;
//                case TRANSFER_TIME_MIN:
//                    this.deviceSelectionPolicy = new FastestDataTransferMinDeviceSelectionPolicy();
//                    break;
//                case TRANSFER_TIME_MAX:
//                    this.deviceSelectionPolicy = new FastestDataTransferMaxDeviceSelectionPolicy();
//                    break;
//                case DATA_LOCALITY:
//                    this.deviceSelectionPolicy = new MoveFewerArgumentsDeviceSelectionPolicy();
//                    break;
//                default:
//                    STREAM_LOGGER.severe("Cannot select a DeviceSelectionPolicy. The selected device selection policy is not valid: " + deviceSelectionPolicyEnum);
//                    throw new GrCUDAException("selected DeviceSelectionPolicy is not valid: " + deviceSelectionPolicyEnum);
//            }
//        } else {
    }

    public GrCUDAStreamPolicy(CUDARuntime runtime,
                              RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum,
                              RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum,
                              DeviceSelectionPolicyEnum deviceSelectionPolicyEnum) {
        this(new GrCUDADevicesManager(runtime), retrieveNewStreamPolicyEnum, retrieveParentStreamPolicyEnum, deviceSelectionPolicyEnum);
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
    private class SingleDeviceSelectionPolicy extends DeviceSelectionPolicy {
        @Override
        Device retrieve(ExecutionDAG.DAGVertex vertex) {
            return devicesManager.getCurrentGPU();
        }
    }

    /**
     * Basic class for multi-GPU scheduling. We simply assign computations to the device with fewer active streams.
     * A stream is active if there's any computation assigned to it that has not been flagged as "finished".
     * The idea is to keep all devices equally busy, and avoid having devices that are used less than others;
     */
    private class StreamAwareDeviceSelectionPolicy extends DeviceSelectionPolicy {

        @Override
        Device retrieve(ExecutionDAG.DAGVertex vertex) {
            return devicesManager.findDeviceWithFewerBusyStreams();
        }
    }
}
