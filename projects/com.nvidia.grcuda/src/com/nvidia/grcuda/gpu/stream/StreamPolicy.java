package com.nvidia.grcuda.gpu.stream;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.GrCUDADevicesManager;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Collectors;

public class StreamPolicy {

    private final RetrieveNewStream retrieveNewStream;
    private final RetrieveParentStream retrieveParentStream;
    private final GrCUDADevicesManager devicesManager;
    private final CUDARuntime runtime;
    private int streamCount;

    public StreamPolicy(RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum, RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum, GrCUDADevicesManager devicesManager, CUDARuntime runtime){
        this.devicesManager = devicesManager;
        this.runtime = runtime;
        switch(retrieveNewStreamPolicyEnum) {
            case FIFO:
                this.retrieveNewStream = new FifoRetrieveStream();
                break;
            case ALWAYS_NEW:
                this.retrieveNewStream = new AlwaysNewRetrieveStream();
                break;
            // case ON_CHEAPEST_DEVICE:
            //     this.retrieveNewStream = new OnCheapestDeviceRetriveStream();
            default:
                this.retrieveNewStream = new FifoRetrieveStream();
        }
        // Get how streams are retrieved for computations with parents;
        switch(retrieveParentStreamPolicyEnum) {
            case DISJOINT:
                this.retrieveParentStream = new DisjointRetrieveParentStream(this.retrieveNewStream);
                break;
            case DEFAULT:
                this.retrieveParentStream = new DefaultRetrieveParentStream();
                break;
            default:
                this.retrieveParentStream = new DefaultRetrieveParentStream();
        }
        this.streamCount = 0;
    }
    /**
     * Create a new {@link CUDAStream} associated to the deviceId and add it to this manager, then return it;
     * @param deviceId
     * @return {@link CUDAStream}
     */
    public CUDAStream createStream(int deviceId) {
        runtime.cudaSetDevice(deviceId);
        CUDAStream newStream = runtime.cudaStreamCreate(this.streamCount);

        devicesManager.addStreamCount(deviceId);
        assert deviceId == newStream.getStreamDeviceId();
        return newStream;
    }

    public void updateStreamCount(){
        this.streamCount++;
    }
    

    
    public CUDAStream getStream(ExecutionDAG.DAGVertex vertex){
        CUDAStream stream;

        if (vertex.isStart()) {
            // Else, if the computation doesn't have parents, provide a new stream to it;
            int cheapestDevice = devicesManager.findCheapestDevice();
            stream = retrieveNewStream.retrieve(cheapestDevice);
        } else {
            // Else, compute the streams used by the parent computations.
            stream = this.retrieveParentStream.retrieve(vertex);
        }
        return stream;
    }

    public void cleanup(){

    }

    public void update(List<CUDAStream> streams){
        retrieveNewStream.update(streams);
    }
    public void update(CUDAStream streams){
        retrieveNewStream.update(streams);
    }


    /**
     * By default, create a new stream every time with respect to the device;
     */
    private class AlwaysNewRetrieveStream extends RetrieveNewStream {

        @Override
        public CUDAStream retrieve(int deviceId) {
            System.out.println("always new retrive called");
            return createStream(deviceId);
        }

    }

    /**
     * Keep a queue of free (currently not utilized) streams for each device, and retrieve the oldest one added to the queue with respect to the device;
     */
    private class FifoRetrieveStream extends RetrieveNewStream{
        /**
         * Keep a queue of free streams;
         */

        @Override
        void update(CUDAStream stream) {
            devicesManager.updateStreams(stream);
        }

        @Override
        void update(Collection<CUDAStream> streamsCollection) {
            devicesManager.updateStreams(streamsCollection);
        }

        @Override
        CUDAStream retrieve(int deviceId) {
            if (!devicesManager.availableStreams(deviceId)) {
                // Create a new stream if none is available;
                System.out.println("line 137 "+deviceId);
                return createStream(deviceId);
            } else {
                // Get the first stream available, and remove it from the list of free streams;
                System.out.println("line 141 "+deviceId);
                CUDAStream stream = devicesManager.retriveStream(deviceId);
                return stream;
            }
        }
    }


    /**
     * By default, use the same stream as the parent computation;
     */
    private static class DefaultRetrieveParentStream extends RetrieveParentStream {

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
                System.out.println("line 190 " + availableParents.get(0).getComputation().getStream().getStreamDeviceId());
                return availableParents.get(0).getComputation().getStream();

            } else {
                // If no parent stream can be reused, provide a new stream to this computation in the same device of the parent
                //   (or possibly a free one, depending on the policy);
                System.out.println("line 196 " + vertex.getComputation().getStream().getStreamDeviceId());
                return retrieveNewStream.retrieve(vertex.getComputation().getStream().getStreamDeviceId());
            }
        }
    }
    /**
     * If a vertex has more than one children, each children is independent (otherwise the dependency would be added
     * from one children to the other, and not from the actual parent).
     * As such, children can be executed on different streams and different devices. If the stream of the parent is free than it is assigned
     * to the child, if the stream of the parent is not available then a new stream is created on the device with the fewest active
     * streams and it is assigned to the child.
     */
    private class DisjointLessBusyRetrieveParentStream extends RetrieveParentStream {
        private final RetrieveNewStream retrieveNewStream;
        // Keep track of computations for which we have already re-used the stream;
        private final Set<ExecutionDAG.DAGVertex> reusedComputations = new HashSet<>();

        public DisjointLessBusyRetrieveParentStream(RetrieveNewStream retrieveNewStream) {
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
                System.out.println("line 183 " + availableParents.get(0).getComputation().getStream().getStreamDeviceId());
                return availableParents.get(0).getComputation().getStream();

            } else {
                // If no parent stream can be reused, provide a new stream to this computation in the same device of the parent
                //   (or possibly a free one, depending on the policy);
                return retrieveNewStream.retrieve(vertex.getComputation().getStream().getStreamDeviceId());
            }
        }
    }
    
}