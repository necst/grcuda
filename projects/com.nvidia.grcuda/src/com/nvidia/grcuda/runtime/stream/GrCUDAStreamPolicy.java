package com.nvidia.grcuda.runtime.stream;

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDALogger;
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.GrCUDADevicesManager;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.oracle.truffle.api.TruffleLogger;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class GrCUDAStreamPolicy {

    /**
     * List of all the {@link CUDAStream} that have been currently allocated;
     */
    protected List<CUDAStream> streams = new ArrayList<>();

    /**
     * Reference to the class that manages the GPU devices in this system;
     */
    private final GrCUDADevicesManager devicesManager;

    private final CUDARuntime runtime;
    private final RetrieveNewStream retrieveNewStream;
    private final RetrieveParentStream retrieveParentStream;
//    private final DeviceSelectionPolicy deviceSelectionPolicy;

    private static final TruffleLogger STREAM_LOGGER = GrCUDALogger.getLogger(GrCUDALogger.STREAM_LOGGER);

    public GrCUDAStreamPolicy(CUDARuntime runtime,
                              GrCUDADevicesManager devicesManager,
                              RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum,
                              RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum) {
        this.runtime = runtime;
        this.devicesManager = devicesManager;
        // Get how streams are retrieved for computations without parents;
        switch (retrieveNewStreamPolicyEnum) {
            case REUSE:
                this.retrieveNewStream = new FifoRetrieveStream();
                break;
            case ALWAYS_NEW:
                this.retrieveNewStream = new AlwaysNewRetrieveStream();
                break;
            default:
                STREAM_LOGGER.severe("Cannot select a RetrieveNewStreamPolicy. The selected execution policy is not valid: " + retrieveNewStreamPolicyEnum);
                throw new GrCUDAException("selected RetrieveNewStreamPolicy is not valid: " + retrieveNewStreamPolicyEnum);
        }
        // Get how streams are retrieved for computations with parents;
        switch (retrieveParentStreamPolicyEnum) {
            case DISJOINT:
                this.retrieveParentStream = new DisjointRetrieveParentStream(this.retrieveNewStream);
                break;
            case SAME_AS_PARENT:
                this.retrieveParentStream = new SameAsParentRetrieveParentStream();
                break;
//            case MULTIGPU_STREAM_AWARE:
//                this.retrieveParentStream = new StreamAwareMultiGPURetrieveParentStream(this.retrieveNewStream);
//                break;
//            case MULTIGPU_DATA_AWARE:
//                this.retrieveParentStream = new DataAwareMultiGPURetrieveParentStream(this.retrieveNewStream);
//                break;
//            case MULTIGPU_DISJOINT_DATA_AWARE:
//                this.retrieveParentStream = new DisjointDataAwareMultiGPURetrieveParentStream(this.retrieveNewStream);
//                break;
            default:
                STREAM_LOGGER.severe("Cannot select a RetrieveParentStreamPolicy. The selected execution policy is not valid: " + retrieveParentStreamPolicyEnum);
                throw new GrCUDAException("selected RetrieveParentStreamPolicy is not valid: " + retrieveParentStreamPolicyEnum);
        }
    }

    public GrCUDAStreamPolicy(CUDARuntime runtime,
                              RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum,
                              RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum) {
        this(runtime, new GrCUDADevicesManager(runtime), retrieveNewStreamPolicyEnum, retrieveParentStreamPolicyEnum);
    }

    /**
     * Create a new {@link CUDAStream} and add it to this manager, then return it;
     */
    public CUDAStream createStream() {
        CUDAStream newStream = runtime.cudaStreamCreate(streams.size());
        streams.add(newStream);
        return newStream;
    }
    
    CUDAStream retrieveNewStream() {
        return this.retrieveNewStream.retrieve();
    }

    CUDAStream retrieveParentStream(ExecutionDAG.DAGVertex vertex) {
        return this.retrieveParentStream.retrieve(vertex);
    }

    /**
     * Update the status of a single stream within the NewStreamRetrieval policy;
     * @param stream a stream to update;
     */
    void updateNewStreamRetrieval(CUDAStream stream) {
        this.retrieveNewStream.update(stream);
    }

    /**
     * Update the status of all streams within the NewStreamRetrieval policy,
     * saying for example that all can be reused;
     */
    void updateNewStreamRetrieval() {
        // All streams are free to be reused;
        this.retrieveNewStream.update(this.streams);
    }

    void cleanupNewStreamRetrieval() {
        this.retrieveNewStream.cleanup();
    }

    /**
     * Obtain the number of streams created so far;
     */
    public int getNumberOfStreams() {
        return this.streams.size();
    }

    public List<CUDAStream> getStreams() {
        return streams;
    }

    public GrCUDADevicesManager getDevicesManager() {
        return devicesManager;
    }

    /**
     * Cleanup and deallocate the streams managed by this manager;
     */
    public void cleanup() {
        this.cleanupNewStreamRetrieval();
        streams.forEach(runtime::cudaStreamDestroy);
        streams.clear();
    }
    
    ////////////////////////////////////////////////
    // List of interfaces that implement policies //
    ////////////////////////////////////////////////

    /**
     * By default, create a new stream every time;
     */
    private class AlwaysNewRetrieveStream extends RetrieveNewStream {

        @Override
        public CUDAStream retrieve() {
            return createStream();
        }
    }

    /**
     * Keep a set of free (currently not utilized) streams, and retrieve one of them instead of always creating new streams;
     */
    private class FifoRetrieveStream extends RetrieveNewStream {

        /**
         * Keep a set of the free available streams;
         */
        private final Set<CUDAStream> uniqueFreeStreams = new HashSet<>();

        @Override
        void update(CUDAStream stream) {
            uniqueFreeStreams.add(stream);
        }

        @Override
        void update(Collection<CUDAStream> streams) {
            uniqueFreeStreams.addAll(streams);
        }

        @Override
        CUDAStream retrieve() {
            if (uniqueFreeStreams.isEmpty()) {
                // Create a new stream if none is available;
                return createStream();
            } else {
                // Get the first stream available, and remove it from the list of free streams;
                CUDAStream stream = uniqueFreeStreams.iterator().next();
                uniqueFreeStreams.remove(stream);
                return stream;
            }
        }
    }

    /**
     * By default, use the same stream as the parent computation;
     */
    private static class SameAsParentRetrieveParentStream extends RetrieveParentStream {

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
     * and computes other streams using the current {@link RetrieveNewStream};
     */
    private static class DisjointRetrieveParentStream extends RetrieveParentStream {
        private final RetrieveNewStream retrieveNewStream;

        // Keep track of computations for which we have already re-used the stream;
        private final Set<ExecutionDAG.DAGVertex> reusedComputations = new HashSet<>();

        public DisjointRetrieveParentStream(RetrieveNewStream retrieveNewStream) {
            this.retrieveNewStream = retrieveNewStream;
        }

        @Override
        public CUDAStream retrieve(ExecutionDAG.DAGVertex vertex) {
            // Keep only parent vertices for which we haven't reused the stream yet;
            List<ExecutionDAG.DAGVertex> availableParents = vertex.getParentVertices().stream()
                    .filter(v -> !reusedComputations.contains(v))
                    .collect(Collectors.toList());
            // If there is at least one stream that can be re-used, take it;
            if (!availableParents.isEmpty()) {
                // The computation cannot be considered again;
                reusedComputations.add(availableParents.get(0));
                // Return the stream associated to this computation;
                return availableParents.get(0).getComputation().getStream();
            } else {
                // If no parent stream can be reused, provide a new stream to this computation
                //   (or possibly a free one, depending on the policy);
                return retrieveNewStream.retrieve();
            }
        }
    }
}
