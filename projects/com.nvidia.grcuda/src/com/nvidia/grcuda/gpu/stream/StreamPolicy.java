package com.nvidia.grcuda.gpu.stream;

import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.GrCUDADevicesManager;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;

public class StreamPolicy {

    private final RetrieveNewStream retrieveNewStream;
    private final RetrieveParentStream retrieveParentStream;
    private ChooseDeviceHeuristic chooseDeviceHeuristic;
    private final GrCUDADevicesManager devicesManager;
    private final CUDARuntime runtime;
    private int streamCount;
    private final StartingVertexPolicy startingVertexPolicy;

    public StreamPolicy(RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum, RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum, GrCUDADevicesManager devicesManager, CUDARuntime runtime) {
        this.devicesManager = devicesManager;
        this.runtime = runtime;
        switch (retrieveNewStreamPolicyEnum) {
            case FIFO:
                this.retrieveNewStream = new FifoRetrieveStream();
                break;
            case ALWAYS_NEW:
            default:
                this.retrieveNewStream = new AlwaysNewRetrieveStream();
        }
        // Get how streams are retrieved for computations with parents;
        switch (retrieveParentStreamPolicyEnum) {
            case DATA_AWARE:
                this.retrieveParentStream = new DataAwareRetrieveParentStream(this.retrieveNewStream);
                break;
            case STREAM_AWARE:
                this.retrieveParentStream = new StreamAwareRetrieveParentStream(this.retrieveNewStream);
                break;
            case DISJOINT_DATA_AWARE:
                this.retrieveParentStream = new DisjointDataAwareRetrieveParentStream(this.retrieveNewStream);
                break;
            case DISJOINT:
                this.retrieveParentStream = new DisjointRetrieveParentStream(this.retrieveNewStream);
                break;
            case DEFAULT:
            default:
                this.retrieveParentStream = new DefaultRetrieveParentStream();
        }
        this.streamCount = 0;
        if (retrieveParentStreamPolicyEnum == RetrieveParentStreamPolicyEnum.DATA_AWARE || retrieveParentStreamPolicyEnum == RetrieveParentStreamPolicyEnum.DISJOINT_DATA_AWARE) {
            this.chooseDeviceHeuristic = new DeviceMoveLessArgument();
        }
        startingVertexPolicy = new StartingVertexPolicy();
    }

    public StreamPolicy(RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum, RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum, ChooseDeviceHeuristicEnum chooseDeviceHeuristicEnum, GrCUDADevicesManager devicesManager, CUDARuntime runtime) {
        this.devicesManager = devicesManager;
        this.runtime = runtime;
        switch (retrieveNewStreamPolicyEnum) {
            case FIFO:
                this.retrieveNewStream = new FifoRetrieveStream();
                break;
            case ALWAYS_NEW:
            default:
                this.retrieveNewStream = new AlwaysNewRetrieveStream();
        }
        // Get how streams are retrieved for computations with parents;
        switch (retrieveParentStreamPolicyEnum) {
            case DATA_AWARE:
                this.retrieveParentStream = new DataAwareRetrieveParentStream(this.retrieveNewStream);
                break;
            case STREAM_AWARE:
                this.retrieveParentStream = new StreamAwareRetrieveParentStream(this.retrieveNewStream);
                break;
            case DISJOINT_DATA_AWARE:
                this.retrieveParentStream = new DisjointDataAwareRetrieveParentStream(this.retrieveNewStream);
                break;
            case DISJOINT:
                this.retrieveParentStream = new DisjointRetrieveParentStream(this.retrieveNewStream);
                break;
            case DEFAULT:
            default:
                this.retrieveParentStream = new DefaultRetrieveParentStream();
        }
        this.streamCount = 0;
        if (retrieveParentStreamPolicyEnum == RetrieveParentStreamPolicyEnum.DATA_AWARE || retrieveParentStreamPolicyEnum == RetrieveParentStreamPolicyEnum.DISJOINT_DATA_AWARE) {
            switch (chooseDeviceHeuristicEnum) {
                case DATA_LOCALITY_NEW:
                    this.chooseDeviceHeuristic = new DeviceMoveLessArgumentNew();
                case TRANSFER_TIME_MIN:
                    this.chooseDeviceHeuristic = new FastestDataTransferMin();
                    break;
                case TRANSFER_TIME_MAX:
                    this.chooseDeviceHeuristic = new FastestDataTransferMax();
                    break;
                case DATA_LOCALITY:
                default:
                    this.chooseDeviceHeuristic = new DeviceMoveLessArgument();
            }
        }
        startingVertexPolicy = new StartingVertexPolicy();
    }

    public GrCUDADevicesManager getDevicesManager() {
        return this.devicesManager;
    }

    /**
     * Create a new {@link CUDAStream} associated to the deviceId and add it to this manager, then return it;
     *
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

    public void updateStreamCount() {
        this.streamCount++;
    }

    public CUDAStream getStream(ExecutionDAG.DAGVertex vertex) {
        CUDAStream stream;

        if (vertex.isStart()) {
            // Else, if the computation doesn't have parents, provide a new stream to it;
            stream = startingVertexPolicy.getStream(vertex);
        } else {
            // Else, compute the streams used by the parent computations.
            stream = this.retrieveParentStream.retrieve(vertex);
        }
        //printPartialGraph(vertex, stream);
        //System.out.println("stream on device: "+stream.getStreamDeviceId());
        return stream;
    }

    public void printPartialGraph(ExecutionDAG.DAGVertex vertex, CUDAStream stream) {
        StringBuilder children = new StringBuilder();
        for (ExecutionDAG.DAGVertex child : vertex.getParentVertices()) {
            children.append(".");
            children.append(child.getId());
        }
        System.out.println(stream.getStreamDeviceId() + "." + vertex.getId() + children.toString());
    }

    public void cleanup() {

    }

    public void update(List<CUDAStream> streams) {
        retrieveNewStream.update(streams);
    }

    public void update(CUDAStream streams) {
        retrieveNewStream.update(streams);
    }

    /**
     * By default, create a new stream every time with respect to the device;
     */
    private class AlwaysNewRetrieveStream extends RetrieveNewStream {
        @Override
        public CUDAStream retrieve(int deviceId) {
            return createStream(deviceId);
        }
    }

    /**
     * Keep a queue of free (currently not utilized) streams for each device, and retrieve the oldest one added to the queue with respect to the device;
     */
    private class FifoRetrieveStream extends RetrieveNewStream {
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
                //System.out.println("line 137 "+deviceId);
                return createStream(deviceId);
            } else {
                // Get the first stream available, and remove it from the list of free streams;
                //System.out.println("line 141 "+deviceId);
                CUDAStream stream = devicesManager.retriveStream(deviceId);
                return stream;
            }
        }
    }

    private class LessBusyRetrieveStream extends RetrieveNewStream {
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
            int cheapestDevice = devicesManager.deviceWithLessActiveStream();
            if (!devicesManager.availableStreams(deviceId)) {
                // Create a new stream if none is available;
                //System.out.println("line 137 "+deviceId);
                return createStream(deviceId);
            } else {
                // Get the first stream available, and remove it from the list of free streams;
                //System.out.println("line 141 "+deviceId);
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
                return availableParents.get(0).getComputation().getStream();
            } else {
                // If no parent stream can be reused, provide a new stream to this computation in the same device of the parent
                //   (or possibly a free one, depending on the policy);
                return retrieveNewStream.retrieve(vertex.getComputation().getStream().getStreamDeviceId());
            }
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
    /* private static class DisjointOrLessActiveRetrieveParentStream extends RetrieveParentStream {
        private final RetrieveNewStream retrieveNewStream;
        // Keep track of computations for which we have already re-used the stream;
        private final Set<ExecutionDAG.DAGVertex> reusedComputations = new HashSet<>();

        public DisjointOrLessActiveRetrieveParentStream(RetrieveNewStream retrieveNewStream) {
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
                // If no parent stream can be reused, provide a new stream to this computation in the same device of the parent
                //   (or possibly a free one, depending on the policy);
                return retrieveNewStream.retrieve(finder.deviceMoveLessArgument(vertex));
            }
        }
    }
    */


    /**
     * If a vertex has more than one children, each children is independent (otherwise the dependency would be added
     * from one children to the other, and not from the actual parent).
     * As such, children can be executed on different streams and different devices. If the stream of the parent is free than it is assigned
     * to the child, if the stream of the parent is not available then a new stream is created on the device with the fewest active
     * streams and it is assigned to the child.
     */
    private class DisjointDataAwareRetrieveParentStream extends RetrieveParentStream {
        private final RetrieveNewStream retrieveNewStream;
        // Keep track of computations for which we have already re-used the stream;
        private final Set<ExecutionDAG.DAGVertex> reusedComputations = new HashSet<>();

        public DisjointDataAwareRetrieveParentStream(RetrieveNewStream retrieveNewStream) {
            this.retrieveNewStream = retrieveNewStream;
        }

        @Override
        public CUDAStream retrieve(ExecutionDAG.DAGVertex vertex) {
            // Keep only parent vertices for which we haven't reused the stream yet;
            List<ExecutionDAG.DAGVertex> availableParentsStream = vertex.getParentVertices().stream()
                    .filter(v -> !reusedComputations.contains(v))
                    .collect(Collectors.toList());
            // If there is at least one stream that can be re-used, take it;
            if (!availableParentsStream.isEmpty()) {
                // The computation cannot be considered again;
                reusedComputations.add(availableParentsStream.get(0));
                // Return the stream associated to this computation;
                return availableParentsStream.get(0).getComputation().getStream();

            } else {
                return retrieveNewStream.retrieve(chooseDeviceHeuristic.getDevice(vertex));
            }
            //return retrieveNewStream.retrieve(finder.deviceMoveLessArgument(vertex));
        }
    }

    private class DataAwareRetrieveParentStream extends RetrieveParentStream {
        private final RetrieveNewStream retrieveNewStream;
        // Keep track of computations for which we have already re-used the stream;
        private final Set<ExecutionDAG.DAGVertex> reusedComputations = new HashSet<>();

        public DataAwareRetrieveParentStream(RetrieveNewStream retrieveNewStream) {
            this.retrieveNewStream = retrieveNewStream;
        }

        @Override
        public CUDAStream retrieve(ExecutionDAG.DAGVertex vertex) {
            // Keep only parent vertices for which we haven't reused the stream yet;
            List<ExecutionDAG.DAGVertex> availableParentsStream = vertex.getParentVertices().stream()
                    .filter(v -> !reusedComputations.contains(v))
                    .collect(Collectors.toList());
            int chosenDevice = chooseDeviceHeuristic.getDevice(vertex);

            if (!availableParentsStream.isEmpty()) {
                for (ExecutionDAG.DAGVertex v : availableParentsStream) {
                    if (v.getComputation().getStream().getStreamDeviceId() == chosenDevice) {
                        reusedComputations.add(v);
                        return v.getComputation().getStream();
                    }
                }
            }

            return retrieveNewStream.retrieve(chooseDeviceHeuristic.getDevice(vertex));
        }
    }

    private class StreamAwareRetrieveParentStream extends RetrieveParentStream {
        private final RetrieveNewStream retrieveNewStream;
        int n = 0;

        public StreamAwareRetrieveParentStream(RetrieveNewStream retrieveNewStream) {
            this.retrieveNewStream = retrieveNewStream;
        }

        @Override
        public CUDAStream retrieve(ExecutionDAG.DAGVertex vertex) {
            //System.out.println("schedule to "+finder.deviceMoveLessArgument(vertex));
            return retrieveNewStream.retrieve(devicesManager.deviceWithLessActiveStream());
        }
    }

    /**
     * Handle the assignment of the stream if the vertex considered is a starting vertex,
     * which is to say that it has no parent computation. It is possible to consider various kind of heuristics can be
     * applied to have the best scheduling, i.e. move less arguments, device with smallest number of running computation.
     */
    private class StartingVertexPolicy {
        // public CUDAStream getStream(ExecutionDAG.DAGVertex vertex){
        //     int cheapestDevice = finder.deviceMoveLessArgument(vertex);
        //     return retrieveNewStream.retrieve(cheapestDevice);
        // }

        public CUDAStream getStream(ExecutionDAG.DAGVertex vertex) {
            return retrieveNewStream.retrieve(devicesManager.deviceWithLessActiveStream());
        }
    }

    private class DeviceMoveLessArgument extends ChooseDeviceHeuristic {
        @Override
        public int getDevice(ExecutionDAG.DAGVertex vertex) {
            long[] argumentSize = new long[devicesManager.getNumberOfGPUs() + 1];
            List<AbstractArray> arguments = vertex.getComputation().getArgumentArray();
            for (AbstractArray a : arguments) {
                if (a.getArrayLocation() == -1) {
                    // last position of the array represents the CPU
                    argumentSize[devicesManager.getNumberOfGPUs()] += a.getSizeBytes();
                } else {
                    argumentSize[a.getArrayLocation()] += a.getSizeBytes();
                }
            }

            //System.out.println("argument for vertex: "+vertex.getId());
            int maxAt = 0;
            for (int i = 0; i < argumentSize.length; i++) {
                maxAt = argumentSize[i] > argumentSize[maxAt] ? i : maxAt;
                //System.out.println("argument size : "+argumentSize[i]+" device id: "+ i);
            }
            if (maxAt == argumentSize.length - 1) {
                return devicesManager.deviceWithLessActiveStream();
            } else {
                return maxAt;
                // return devicesManager.deviceWithLessActiveStream();
            }
        }
    }

    private class DeviceMoveLessArgumentNew extends ChooseDeviceHeuristic {
        @Override
        public int getDevice(ExecutionDAG.DAGVertex vertex) {
            long[] presentArgumentSize = new long[devicesManager.getNumberOfGPUs()+1];
            List<AbstractArray> arguments = vertex.getComputation().getArgumentArray();
            for(AbstractArray a : arguments){
                for(int location : a.getArrayLocations()){
                    if(location == -1){
                        // last position of the array represents the CPU
                        presentArgumentSize[devicesManager.getNumberOfGPUs()] += a.getSizeBytes();
                    }else{
                        presentArgumentSize[location] += a.getSizeBytes();
                    }
                }
            }

            //System.out.println("argument for vertex: "+vertex.getId());
            int maxAt = 0;
            for (int i = 0; i < presentArgumentSize.length; i++) {
                maxAt = presentArgumentSize[i] > presentArgumentSize[maxAt] ? i : maxAt;
                //System.out.println("argument size : "+argumentSize[i]+" device id: "+ i);
            }

            if(maxAt == presentArgumentSize.length-1){
                return devicesManager.deviceWithLessActiveStream();
            }else{
                return maxAt;
                // return devicesManager.deviceWithLessActiveStream();
            }
        }
    }

    // new
    private class FastestDataTransferMax extends ChooseDeviceHeuristic {

        private double[][] linkBandwidth = new double[devicesManager.getNumberOfGPUs() + 1][devicesManager.getNumberOfGPUs() + 1];

        public FastestDataTransferMax() {
            List<List<String>> records = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(new FileReader("connection_graph.csv"))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] values = line.split(",");
                    records.add(Arrays.asList(values));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            //System.out.println(records);

            for (int il = 1; il<records.size(); il++) {
                if (Integer.parseInt(records.get(il).get(0)) != -1) {
                    this.linkBandwidth[Integer.parseInt(records.get(il).get(0))][Integer.parseInt(records.get(il).get(1))] = Double.parseDouble(records.get(il).get(2));
                } else {
                    this.linkBandwidth[2][Integer.parseInt(records.get(il).get(1))] = Double.parseDouble(records.get(il).get(2));
                    this.linkBandwidth[Integer.parseInt(records.get(il).get(1))][2] = Double.parseDouble(records.get(il).get(2));
                }
            }
        }

        @Override
        public int getDevice(ExecutionDAG.DAGVertex vertex) {
            // last position of the arrays represents the CPU
            long[] presentArgumentSize = new long[devicesManager.getNumberOfGPUs() + 1];
            double[] argumentTransferTime = new double[devicesManager.getNumberOfGPUs() + 1];
            List<AbstractArray> arguments = vertex.getComputation().getArgumentArray();

            double band;
            // for each argument array
            for (AbstractArray a : arguments) {
                // check all available GPUs and compute the tentative transfer time for each of them
                for (int i = 0; i < presentArgumentSize.length - 1; i++) {
                    // hypoteses: we consider the max bandwith towards the device i
                    // initialization: min possible value
                    band = 0.0;
                    // if array a already present in device i, band is infinity
                    if (a.getArrayLocations().contains(i)) {
                        band = Double.POSITIVE_INFINITY;
                        // else we consider the max band to move array a to device i
                    } else {
                        // from each possible location containing the array a
                        for (int location : a.getArrayLocations()) {
                            band = linkBandwidth[location][i] > band ? linkBandwidth[location][i] : band;
                        }
                    }
                    argumentTransferTime[i] += a.getSizeBytes() / band;
                }
            }

            //System.out.println("argument for vertex: "+vertex.getId());
            // best device has minimum transfer time
            int minAt = 0;
            for (int i = 0; i < argumentTransferTime.length - 1; i++) {
                minAt = argumentTransferTime[i] < argumentTransferTime[minAt] ? i : minAt;
                //System.out.println("argument size : "+argumentSize[i]+" device id: "+ i);
            }
            return minAt;
        }
    }

    private class FastestDataTransferMin extends ChooseDeviceHeuristic {

        private double[][] linkBandwidth = new double[devicesManager.getNumberOfGPUs() + 1][devicesManager.getNumberOfGPUs() + 1];

        public FastestDataTransferMin () {
            List<List<String>> records = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(new FileReader("connection_graph.csv"))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] values = line.split(",");
                    records.add(Arrays.asList(values));
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
            //System.out.println(records);

            for (int il = 1; il<records.size(); il++) {
                if (Integer.parseInt(records.get(il).get(0)) != -1) {
                    this.linkBandwidth[Integer.parseInt(records.get(il).get(0))][Integer.parseInt(records.get(il).get(1))] = Double.parseDouble(records.get(il).get(2));
                } else {
                    this.linkBandwidth[2][Integer.parseInt(records.get(il).get(1))] = Double.parseDouble(records.get(il).get(2));
                    this.linkBandwidth[Integer.parseInt(records.get(il).get(1))][2] = Double.parseDouble(records.get(il).get(2));
                }
            }
        }

        @Override
        public int getDevice(ExecutionDAG.DAGVertex vertex) {
            // last position of the arrays represents the CPU
            long[] presentArgumentSize = new long[devicesManager.getNumberOfGPUs() + 1];
            double[] argumentTransferTime = new double[devicesManager.getNumberOfGPUs() + 1];
            List<AbstractArray> arguments = vertex.getComputation().getArgumentArray();

            double band;
            // for each argument array
            for (AbstractArray a :arguments) {
                // check all available GPUs and compute the tentative transfer time for each of them
                for (int i = 0; i < presentArgumentSize.length - 1; i++) {
                    // hypotesis: we consider the min bandwidth towards the device i
                    // initialization: max possible value
                    band = Double.POSITIVE_INFINITY;
                    // if array a already present in device i, band is infinity
                    // else we consider the min band to move array a to device i
                    if (!a.getArrayLocations().contains(i)) {
                        // from each possible location containing the array a
                        for (int location : a.getArrayLocations()) {
                            band = linkBandwidth[location][i] < band ? linkBandwidth[location][i] : band;
                        }
                    }
                    argumentTransferTime[i] += a.getSizeBytes() / band;
                }
            }

            //System.out.println("argument for vertex: "+vertex.getId());
            // best device has minimum transfer time
            int minAt = 0;
            for(int i = 0; i<argumentTransferTime.length-1; i++) {
                minAt = argumentTransferTime[i] < argumentTransferTime[minAt] ? i : minAt;
                //System.out.println("argument size : "+argumentSize[i]+" device id: "+ i);
            }
            return minAt;
        }
    }
}