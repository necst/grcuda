package com.nvidia.grcuda.gpu.stream;

import com.nvidia.grcuda.CUDAEvent;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Collectors;

public class GrCUDAStreamManager {

    /**
     * List of {@link CUDAStream} that have been currently allocated;
     */
    protected List<CUDAStream> streams = new ArrayList<>();
    /**
     * Reference to the CUDA runtime that manages the streams;
     */
    protected final CUDARuntime runtime;
    /**
     * Track the active computations each stream has, excluding the default stream;
     */
    protected final Map<CUDAStream, Set<GrCUDAComputationalElement>> activeComputationsPerStream = new HashMap<>();

    private final RetrieveNewStream retrieveNewStream;
    private final RetrieveParentStream retrieveParentStream;

    public GrCUDAStreamManager(CUDARuntime runtime) { 
        this(runtime, runtime.getContext().getRetrieveNewStreamPolicy(), runtime.getContext().getRetrieveParentStreamPolicyEnum());
    }

    public GrCUDAStreamManager(
            CUDARuntime runtime,
            RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum,
            RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum) {
        this.runtime = runtime;
        // Get how streams are retrieved for computations without parents;
        switch(retrieveNewStreamPolicyEnum) {
            case FIFO:
                this.retrieveNewStream = new FifoRetrieveStream();
                break;
            case ALWAYS_NEW:
                this.retrieveNewStream = new AlwaysNewRetrieveStream();
                break;
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
    }

    /**
     * Assign a {@link CUDAStream} to the input computation, based on its dependencies and on the available streams.
     * This function has no effect if the stream was manually specified by the user;
     * @param vertex an input computation for which we want to assign a stream
     */
    public void assignStream(ExecutionDAG.DAGVertex vertex) {

        // If the computation cannot use customized streams, return immediately;
        if (vertex.getComputation().canUseStream()) {
            CUDAStream stream;
            if (vertex.isStart()) {
                // Else, if the computation doesn't have parents, provide a new stream to it;
                stream = retrieveNewStream.retrieve();
            } else {
                // Else, compute the streams used by the parent computations.
                stream = this.retrieveParentStream.retrieve(vertex);
            }
            // Set the stream;
            vertex.getComputation().setStream(stream);
            // Update the computation counter;
            addActiveComputation(vertex.getComputation());
            // Associate all the arrays in the computation to the selected stream,
            //   to enable CPU accesses on managed memory arrays currently not being used by the GPU.
            // This is required as on pre-Pascal GPUs all unified memory pages are locked by the GPU while code is running on the GPU,
            //   even if the GPU is not using some of the pages. Enabling memory-stream association allows the CPU to
            //   access memory not being currently used by a kernel;
            vertex.getComputation().associateArraysToStream();
        }
    }

//    /**
//     * Given a computation, synchronize all its parent streams. The caller thread will be blocked until all the
//     * computations on the parents streams are finished;
//     * @param vertex a computation whose parents should be synchronized
//     */
//    public void syncParentStreamsOld(ExecutionDAG.DAGVertex vertex) {
//
//        // Skip syncing if no computation is active;
//        if (this.isAnyComputationActive()) {
//            Set<GrCUDAComputationalElement> computationsToSync = new HashSet<>(vertex.getParentComputations());
//
//            // Retrieve an additional stream dependency from the computation, if required;
//            Optional<CUDAStream> additionalStream = vertex.getComputation().additionalStreamDependency();
//            if (additionalStream.isPresent()) {
//                CUDAStream stream = additionalStream.get();
//                // If we require synchronization on the default stream, perform it in a specialized way;
//                if (stream.isDefaultStream()) {
//                    System.out.println("--\tsync stream " + stream + " by " + vertex.getComputation());
//                    // Synchronize the device;
//                    runtime.cudaDeviceSynchronize();
//                    // All computations are now finished;
//                    resetActiveComputationState();
//                } else if (!getParentStreams(computationsToSync).contains(stream)) {
//                    // Else add the computations related to the additional streams to the set and sync it,
//                    //   as long as the additional stream isn't the same as the one that we have to sync anyway;
//                    System.out.println("--\tsyncing additional stream " + stream + "...");
//                    computationsToSync.addAll(activeComputationsPerStream.get(stream));
//                    syncParentStreamsImpl(computationsToSync, vertex.getComputation());
//                } else {
//                    syncParentStreamsImpl(computationsToSync, vertex.getComputation());
//                }
//            } else {
//                syncParentStreamsImpl(computationsToSync, vertex.getComputation());
//            }
//        }
//    }

    public void syncParentStreams(ExecutionDAG.DAGVertex vertex) {
        // If the vertex can be executed on a CUDA stream, use CUDA events,
        //   otherwise use stream/device synchronization to block the host until synchronization is done;
        if (vertex.getComputation().canUseStream()) {
            syncStreamsUsingEvents(vertex);
        } else {
            if (this.isAnyComputationActive()) {
                List<GrCUDAComputationalElement> computationsToSync = vertex.getParentComputations();
                Set<CUDAStream> streamsToSync = getParentStreams(computationsToSync);
                Optional<CUDAStream> additionalStream = vertex.getComputation().additionalStreamDependency();
                if (additionalStream.isPresent()) {
                    CUDAStream stream = additionalStream.get();
                    // If we require synchronization on the default stream, perform it in a specialized way;
                    if (stream.isDefaultStream()) {
                        System.out.println("--\tsync stream " + stream + " by " + vertex.getComputation());
                        // Synchronize the device;
                        runtime.cudaDeviceSynchronize();
                        // All computations are now finished;
                        resetActiveComputationState();
                    } else if (!streamsToSync.contains(stream)) {
                        // Else add the computations related to the additional streams to the set and sync it,
                        //   as long as the additional stream isn't the same as the one that we have to sync anyway;
                        System.out.println("--\tsyncing additional stream " + stream + "...");
                        streamsToSync.add(stream);
                        syncParentStreamsImpl(streamsToSync, computationsToSync, vertex.getComputation());
                    } else {
                        syncParentStreamsImpl(streamsToSync, computationsToSync, vertex.getComputation());
                    }
                } else {
                    syncParentStreamsImpl(streamsToSync, computationsToSync, vertex.getComputation());
                }
            }
        }
    }

    /**
     * Obtain the set of CUDAStreams that have to be synchronized;
     * @param computationsToSync a set of computations to sync
     * @return the set of CUDAStreams that have to be synchronized
     */
    private Set<CUDAStream> getParentStreams(Collection<GrCUDAComputationalElement> computationsToSync) {
        return computationsToSync.stream().map(GrCUDAComputationalElement::getStream).collect(Collectors.toSet());
    }

    /**
     * If a computation can be scheduled on a stream, use {@link CUDAEvent} to synchronize parent computations
     * without blocking the CPU host
     * @param vertex a computation whose parent's streams must be synchronized
     */
    private void syncStreamsUsingEvents(ExecutionDAG.DAGVertex vertex) {
        Set<CUDAStream> streamToSync = getParentStreams(vertex.getParentComputations());
        for (CUDAStream stream : streamToSync) {
            // Skip synchronization on the same stream where the new computation is executed,
            //   as operations scheduled on a stream are executed in order;
            if (!vertex.getComputation().getStream().equals(stream)) {
                // Create a new synchronization event on the stream;
                CUDAEvent event = runtime.cudaEventCreate();
                runtime.cudaEventRecord(event, stream);
                runtime.cudaStreamWaitEvent(vertex.getComputation().getStream(), event);

                System.out.println("\t* wait event on stream; stream to sync=" + stream.getStreamNumber()
                        + "; stream that waits=" + vertex.getComputation().getStream().getStreamNumber()
                        + "; event=" + event.getEventNumber());
            }
        }
    }

//    /**
//     * Synchronize a list of computations on their streams;
//     * @param computationsToSync a list of computations whose streams should be synced
//     * @param computationThatSyncs the computation that triggered the syncing process
//     */
//    private void syncParentStreamsImpl(
//            Set<GrCUDAComputationalElement> computationsToSync,
//            GrCUDAComputationalElement computationThatSyncs) {
//
//        // Keep a set of streams synced so far, to avoid syncing the same stream multiple times;
//        Set<CUDAStream> syncedStreams = new HashSet<>();
//
//        computationsToSync.forEach(c -> {
//            // When scheduling a computation that uses the same stream of the parent, avoid manual synchronization as it is handled by CUDA;
//            if (!(computationThatSyncs.canUseStream() && computationThatSyncs.getStream().equals(c.getStream()))) {
//                // Synchronize computations that are not yet finished and can use streams;
//                if (c.canUseStream() && !c.isComputationFinished()) {
//                    if (!syncedStreams.contains(c.getStream())) {
//                        System.out.println("--\tsync stream " + c.getStream() + " by " + computationThatSyncs);
//                        runtime.cudaStreamSynchronize(c.getStream());
//                        syncedStreams.add(c.getStream());
//                    }
//                    // Set the parent computations as finished;
//                    c.setComputationFinished();
//                    // Decrement the active computation count;
//                    removeActiveComputation(c);
//                }
//            } else {
//                System.out.println("--\tavoid manual sync of " + c + " by " + computationThatSyncs);
//            }
//        });
//    }

    private void syncParentStreamsImpl(Set<CUDAStream> streamsToSync,
                                       List<GrCUDAComputationalElement> parentComputations,
                                       GrCUDAComputationalElement computationThatSyncs) {
        // Synchronize streams;
        streamsToSync.forEach(s -> {
            System.out.println("--\tsync stream=" + s.getStreamNumber() + " by " + computationThatSyncs);
            runtime.cudaStreamSynchronize(s);
        });
        // Book-keeping: all computations are now guaranteed to be finished;
        parentComputations.forEach(c -> {
            // Set the parent computations as finished;
            c.setComputationFinished();
            // Decrement the active computation count;
            removeActiveComputation(c);
        });
    }

    /**
     * Create a new {@link CUDAStream} and add it to this manager, then return it;
     */
    public CUDAStream createStream() {
        CUDAStream newStream = runtime.cudaStreamCreate(streams.size());
        streams.add(newStream);
        return newStream;
    }

    public void syncStream(CUDAStream stream) {
        runtime.cudaStreamSynchronize(stream);
    }

    /**
     * Obtain the number of streams managed by this manager;
     */
    public int getNumberOfStreams() {
        return streams.size();
    }

    public int getNumActiveComputationsOnStream(CUDAStream stream) {
        return activeComputationsPerStream.get(stream).size();
    }

    /**
     * Check if any computation is currently marked as active, and is running on a stream.
     * If so, scheduling of new computations is likely to require synchronizations of some sort;
     * @return if any computation is considered active on a stream
     */
    public boolean isAnyComputationActive() { return !this.activeComputationsPerStream.isEmpty(); }

    protected void addActiveComputation(GrCUDAComputationalElement computation) {
        CUDAStream stream = computation.getStream();
        // Start tracking the stream if it wasn't already tracked;
        if (!activeComputationsPerStream.containsKey(stream)) {
            activeComputationsPerStream.put(stream, new HashSet<>());
        }
        // Associate the computation to the stream;
        activeComputationsPerStream.get(stream).add(computation);
    }

    /**
     * Remove a computation from the map that associates streams to their active computations,
     * and mark the stream as free if no other computations are active on the stream;
     * @param computation a computation that is no longer active
     */
    protected void removeActiveComputation(GrCUDAComputationalElement computation) {
        CUDAStream stream = computation.getStream();
        activeComputationsPerStream.get(stream).remove(computation);
        // If this stream doesn't have any computation associated to it, it's free to use;
        if (activeComputationsPerStream.get(stream).isEmpty()) {
            activeComputationsPerStream.remove(stream);
            retrieveNewStream.update(stream);
        }
    }

    /**
     * Reset the association between streams and computations. All computations are finished, and all streams are free;
     */
    private void resetActiveComputationState() {
        activeComputationsPerStream.keySet().forEach(s ->
            activeComputationsPerStream.get(s).forEach(GrCUDAComputationalElement::setComputationFinished)
        );
        // Streams don't have any active computation;
        activeComputationsPerStream.clear();
        // All streams are free;
        retrieveNewStream.update(streams);
    }

    /**
     * Cleanup and deallocate the streams managed by this manager;
     */
    public void cleanup() {
        streams.forEach(runtime::cudaStreamDestroy);
        activeComputationsPerStream.clear();
        retrieveNewStream.cleanup();
        streams.clear();
    }

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
     * Keep a queue of free (currently not utilized) streams, and retrieve the oldest one added to the queue;
     */
    private class FifoRetrieveStream extends RetrieveNewStream {

        /**
         * Keep a queue of free streams;
         */
        private final Queue<CUDAStream> freeStreams = new ArrayDeque<>();

        @Override
        void update(CUDAStream stream) {
            freeStreams.add(stream);
        }

        @Override
        void update(Collection<CUDAStream> streams) {
            freeStreams.addAll(streams);
        }

        @Override
        CUDAStream retrieve() {
            if (freeStreams.isEmpty()) {
                // Create a new stream if none is available;
                return createStream();
            } else {
                // Get the first stream available, and remove it from the list of free streams;
                return freeStreams.poll();
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
                return availableParents.get(0).getComputation().getStream();
            } else {
                // If no parent stream can be reused, provide a new stream to this computation
                //   (or possibly a free one, depending on the policy);
                return retrieveNewStream.retrieve();
            }
        }
    }
}
