package com.nvidia.grcuda.test.runtime.stream;

import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import com.nvidia.grcuda.test.util.mock.ArgumentMock;
import com.nvidia.grcuda.test.util.mock.DeviceArrayMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAExecutionContextMockBuilder;
import com.nvidia.grcuda.test.util.mock.KernelExecutionMock;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class MultiGPUExecutionDAGMockTest {

    @Parameterized.Parameters
    public static Collection<Object[]> data() {

        return GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                {RetrieveNewStreamPolicyEnum.ALWAYS_NEW, RetrieveNewStreamPolicyEnum.REUSE}
        }));
    }

    private final RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy;

    public MultiGPUExecutionDAGMockTest(RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy) {
        this.retrieveNewStreamPolicy = retrieveNewStreamPolicy;
    }

    /**
     * Schedule for execution a sequence of mock GrCUDAComputationalElement,
     * and validate that the GPU scheduling of the computation is the one expected.
     * @param computations a sequence of computations to be scheduled
     * @param gpuScheduling a list of gpu identifiers. Each identifier "i" represents the GPU scheduling for the i-th computation;
     * @throws UnsupportedTypeException
     */
    public static void executeMockComputationAndValidate(List<GrCUDAComputationalElement> computations, List<Integer> gpuScheduling) throws UnsupportedTypeException {
        assertEquals(computations.size(), gpuScheduling.size());
        for (int i = 0; i < computations.size(); i++) {
            GrCUDAComputationalElement c = computations.get(i);
            c.schedule();
            int expected = gpuScheduling.get(i);
            int actual = c.getStream().getStreamDeviceId();
            if (expected != actual) {
                System.out.println("wrong GPU allocation for kernel " + i + "=" + c + "; expected=" + expected + "; actual=" + actual);
            }
            assertEquals(expected, actual);
        }
    }

    // K0 -> K4 -> K8 ---> K10
    // K1 -> K5 /     \--> K11
    // K2 -> K6 -> K9 -\-> K12
    // K3 -> K7 /------\-> K13
    public static List<GrCUDAComputationalElement> manyKernelsMockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a = new DeviceArrayMock(10);
        DeviceArrayMock b = new DeviceArrayMock(10);
        DeviceArrayMock c = new DeviceArrayMock(10);
        DeviceArrayMock d = new DeviceArrayMock(10);
        return Arrays.asList(
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a))),
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(b))),
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(c))),
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(d))),

            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a))),
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(b))),
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(c))),
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(d))),

            new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(b))),
            new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(c), new ArgumentMock(d))),

            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a, true))),
            // When using stream-aware and 4 GPUs, this is scheduled on device 2 (of 4) as device 1 has synced the computation on it (with K8),
            // and device 2 is the first device with fewer streams;
            new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a, true), new ArgumentMock(b))),
            // When using stream-aware and 4 GPUs, this is scheduled on device 3 (reuse the stream of K9);
            new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a, true), new ArgumentMock(c))),
            // When using stream-aware and 4 GPUs, this is scheduled on device 2 (device with fewer streams, device 1 has 2);
            new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a, true), new ArgumentMock(d)))
        );
    }

    // Simply schedule 10 kernels on independent data;
    public static List<GrCUDAComputationalElement> manyIndependentKernelsMockComputation(AsyncGrCUDAExecutionContext context) {
        return Arrays.asList(
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10))))
        );
    }

    // (Ar) --> (A, B) --> (A, B, C) -> (A, B, D)
    // (Br) -/         /             /
    // (Cr) ----------/             /
    // (Dr) -----------------------/
    public static List<GrCUDAComputationalElement> joinPipelineMockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a = new DeviceArrayMock(10);
        DeviceArrayMock b = new DeviceArrayMock(10);
        DeviceArrayMock c = new DeviceArrayMock(10);
        DeviceArrayMock d = new DeviceArrayMock(100);
        return Arrays.asList(
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(b, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(c, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(d, true))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(b))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(b), new ArgumentMock(c))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(b), new ArgumentMock(d)))
        );
    }

    // (A) --> (Ar, B) --> (A, B, C)
    // (B) -/           /
    // (C) --> (C, D) -/
    // (D) -/
    public static List<GrCUDAComputationalElement> joinPipeline2MockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a = new DeviceArrayMock(10);
        DeviceArrayMock b = new DeviceArrayMock(10);
        DeviceArrayMock c = new DeviceArrayMock(10);
        DeviceArrayMock d = new DeviceArrayMock(100);
        return Arrays.asList(
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(b, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(c, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(d, true))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a, true), new ArgumentMock(b))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(c), new ArgumentMock(d))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(b), new ArgumentMock(c)))
        );
    }

    // (A) --> (Ar, B) --> (B, C)
    // (B) -/           /
    // (C) --> (C, D) -/
    // (D) -/
    public static List<GrCUDAComputationalElement> joinPipeline3MockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a = new DeviceArrayMock(10);
        DeviceArrayMock b = new DeviceArrayMock(10);
        DeviceArrayMock c = new DeviceArrayMock(10);
        DeviceArrayMock d = new DeviceArrayMock(100);
        return Arrays.asList(
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(b, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(c, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(d, true))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a, true), new ArgumentMock(b))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(c), new ArgumentMock(d))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(b), new ArgumentMock(c)))
        );
    }

    // (A) --> (Ar, B) -> (A, C, D) -> (A, C)
    // (B) -/          /
    // (C) -----------/
    // (D) ----------/
    public static List<GrCUDAComputationalElement> joinPipeline4MockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a = new DeviceArrayMock(10);
        DeviceArrayMock b = new DeviceArrayMock(10);
        DeviceArrayMock c = new DeviceArrayMock(10);
        DeviceArrayMock d = new DeviceArrayMock(100);
        return Arrays.asList(
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(b, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(c, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(d, true))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a, true), new ArgumentMock(b))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(c), new ArgumentMock(d))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(c)))
        );
    }

    // (X) --> (Z) --> (A)
    // (Y) -/      \-> (B)
    public static List<GrCUDAComputationalElement> forkJoinMockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a1 = new DeviceArrayMock(10);
        DeviceArrayMock a2 = new DeviceArrayMock(10);
        return Arrays.asList(
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a1))),
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a2))),
            new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a1), new ArgumentMock(a2))),
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a1))),
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a2)))
        );
    }

    private final static int IMAGE_NUM_STREAMS = 4;
    private final static int HITS_NUM_STREAMS = 2;

    @Test
    public void deviceSelectionAlwaysOneImageTest() throws UnsupportedTypeException {
        // Test that no matter how many GPU we have, the SINGLE_GPU policy always selects the number 0;
        for (int i = 1; i < 4; i++) {
            AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                    .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                    .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                    .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.SINGLE_GPU)
                    .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                    .setNumberOfGPUsToUse(i).setNumberOfAvailableGPUs(i).build();
            ComplexExecutionDAGMockTest.executeMockComputation(ComplexExecutionDAGMockTest.imageMockComputation(context));
            context.getDeviceList().forEach(d -> d.getStreams().forEach(s -> assertEquals(0, s.getStreamDeviceId())));
            assertEquals(IMAGE_NUM_STREAMS, context.getStreamManager().getNumberOfStreams());
            assertEquals(IMAGE_NUM_STREAMS, context.getStreamManager().getDevice(0).getNumberOfFreeStreams());
            assertEquals(0, context.getStreamManager().getDevice(0).getNumberOfBusyStreams());
            assertEquals(IMAGE_NUM_STREAMS, context.getStreamManager().getDevice(0).getStreams().size());
        }
    }

    @Test
    public void lessBusyWithOneGPUImageTest() throws UnsupportedTypeException {
        // Test that no matter how many GPU we have, the STREAM_AWARE policy with just 1 GPU always selects the number 0;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.STREAM_AWARE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(1).setNumberOfAvailableGPUs(1).build();
        ComplexExecutionDAGMockTest.executeMockComputation(ComplexExecutionDAGMockTest.imageMockComputation(context));
        context.getDeviceList().forEach(d -> d.getStreams().forEach(s -> assertEquals(0, s.getStreamDeviceId())));
        assertEquals(IMAGE_NUM_STREAMS, context.getStreamManager().getNumberOfStreams());
        assertEquals(IMAGE_NUM_STREAMS, context.getStreamManager().getDevice(0).getNumberOfFreeStreams());
        assertEquals(0, context.getStreamManager().getDevice(0).getNumberOfBusyStreams());
        assertEquals(IMAGE_NUM_STREAMS, context.getStreamManager().getDevice(0).getStreams().size());
    }

    @Test
    public void deviceSelectionAlwaysOneHitsTest() throws UnsupportedTypeException {
        // Test that no matter how many GPU we have, the SINGLE_GPU policy always selects the number 0;
        for (int i = 1; i < 4; i++) {
            AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                    .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                    .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                    .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.SINGLE_GPU)
                    .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                    .setNumberOfGPUsToUse(i).setNumberOfAvailableGPUs(i).build();
            ComplexExecutionDAGMockTest.executeMockComputation(ComplexExecutionDAGMockTest.hitsMockComputation(context));
            context.getDeviceList().forEach(d -> d.getStreams().forEach(s -> assertEquals(0, s.getStreamDeviceId())));
            assertEquals(HITS_NUM_STREAMS, context.getStreamManager().getNumberOfStreams());
            assertEquals(HITS_NUM_STREAMS, context.getStreamManager().getDevice(0).getNumberOfFreeStreams());
            assertEquals(0, context.getStreamManager().getDevice(0).getNumberOfBusyStreams());
            assertEquals(HITS_NUM_STREAMS, context.getStreamManager().getDevice(0).getStreams().size());
        }
    }

    @Test
    public void lessBusyWithOneGPUHitsTest() throws UnsupportedTypeException {
        // Test that no matter how many GPU we have, the STREAM_AWARE policy with just 1 GPU always selects the number 0;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.STREAM_AWARE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(1).setNumberOfAvailableGPUs(1).build();
        ComplexExecutionDAGMockTest.executeMockComputation(ComplexExecutionDAGMockTest.hitsMockComputation(context));
        context.getDeviceList().forEach(d -> d.getStreams().forEach(s -> assertEquals(0, s.getStreamDeviceId())));
        assertEquals(HITS_NUM_STREAMS, context.getStreamManager().getNumberOfStreams());
        assertEquals(HITS_NUM_STREAMS, context.getStreamManager().getDevice(0).getNumberOfFreeStreams());
        assertEquals(0, context.getStreamManager().getDevice(0).getNumberOfBusyStreams());
        assertEquals(HITS_NUM_STREAMS, context.getStreamManager().getDevice(0).getStreams().size());
    }

    // Test the STREAM_AWARE policy on 2 and 3 GPUs, on the image pipeline and HITS DAGs.
    // In each case, validate the mapping of each computation on the right GPUs,
    // and the total number of streams created;

    @Test
    public void lessBusyWithTwoGPUImageTest() throws UnsupportedTypeException {
        int numGPU = 2;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.STREAM_AWARE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(ComplexExecutionDAGMockTest.imageMockComputation(context),
                Arrays.asList(
                        0, 1, 0,
                        0, 1,
                        0, 1,
                        0, 0, 1, 0, 0));
        assertEquals(IMAGE_NUM_STREAMS, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void lessBusyWithThreeGPUImageTest() throws UnsupportedTypeException {
        int numGPU = 3;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.STREAM_AWARE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(ComplexExecutionDAGMockTest.imageMockComputation(context),
                Arrays.asList(
                        0, 1, 2,
                        0, 1,
                        2, 0,
                        2, 2, 1, 0, 0));
        assertEquals(IMAGE_NUM_STREAMS, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void lessBusyWithTwoGPUHitsTest() throws UnsupportedTypeException {
        int numGPU = 2;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.STREAM_AWARE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(ComplexExecutionDAGMockTest.hitsMockComputation(context),
                Arrays.asList(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0));
        assertEquals(HITS_NUM_STREAMS, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void lessBusyWithThreeGPUHitsTest() throws UnsupportedTypeException {
        int numGPU = 3;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.STREAM_AWARE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        // Same as 2 GPUs, it never makes sense to use the 3rd GPU;
        executeMockComputationAndValidate(ComplexExecutionDAGMockTest.hitsMockComputation(context),
                Arrays.asList(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0));
        assertEquals(HITS_NUM_STREAMS, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void lessBusyManyKernelsWithFourGPUTest() throws UnsupportedTypeException {
        int numGPU = 4;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.STREAM_AWARE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(manyKernelsMockComputation(context),
                Arrays.asList(0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 0, 0, 2, 1));
        assertEquals(6, context.getStreamManager().getNumberOfStreams());
    }

    // (X) --> (Z) --> (A)
    // (Y) -/      \-> (B)
    @Test
    public void lessBusyForkJoinWithTwoGPUTest() throws UnsupportedTypeException {
        int numGPU = 2;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.STREAM_AWARE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        // FIXME: When using stream-aware and 2 GPUs, the 5th kernel should be scheduled on device 2 as device 1 has synced the computation on it,
        //  and device 2 is the first device with fewer streams active (0, in this case).
        //  Currently this does not happen, because we cannot know if the computation on device 2 is actually over when we do the scheduling,
        //  although this does not affect correctness.
        executeMockComputationAndValidate(forkJoinMockComputation(context),
                Arrays.asList(0, 1, 0, 0, 0));
        assertEquals(3, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void roundRobinTest() throws UnsupportedTypeException {
        int[] gpus = {1, 4, 8};
        for (int numGPU : gpus) {
            AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                    .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                    .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                    .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.ROUND_ROBIN)
                    .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                    .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
            executeMockComputationAndValidate(manyIndependentKernelsMockComputation(context),
                    Arrays.asList(0, 1 % numGPU, 2 % numGPU, 3 % numGPU, 4 % numGPU, 5 % numGPU, 6 % numGPU, 7 % numGPU, 8 % numGPU, 9 % numGPU));
            assertEquals(10, context.getStreamManager().getNumberOfStreams());
        }
    }

    @Test
    public void roundRobinWithThreeGPUImageTest() throws UnsupportedTypeException {
        int numGPU = 3;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.ROUND_ROBIN)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(ComplexExecutionDAGMockTest.imageMockComputation(context),
                Arrays.asList(
                        0, 1, 2,
                        0, 1,
                        2, 0,
                        2, 2, 1, 0, 0));
        assertEquals(IMAGE_NUM_STREAMS, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void roundRobinWithFourGPUHitsTest() throws UnsupportedTypeException {
        int numGPU = 4;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.ROUND_ROBIN)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(ComplexExecutionDAGMockTest.hitsMockComputation(context),
                Arrays.asList(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0));
        assertEquals(HITS_NUM_STREAMS, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void roundRobinManyKernelsWithFourGPUTest() throws UnsupportedTypeException {
        int numGPU = 4;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.STREAM_AWARE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(manyKernelsMockComputation(context),
                Arrays.asList(0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 0, 0, 2, 1));
        assertEquals(6, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void roundRobinForkJoinWithTwoGPUTest() throws UnsupportedTypeException {
        int numGPU = 2;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.ROUND_ROBIN)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(forkJoinMockComputation(context),
                Arrays.asList(0, 1, 0, 0, 0));
        assertEquals(3, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void minTransferTest() throws UnsupportedTypeException {
        int[] gpus = {1, 4, 8};
        for (int numGPU : gpus) {
            AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                    .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                    .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                    .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                    .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                    .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
            executeMockComputationAndValidate(manyIndependentKernelsMockComputation(context),
                    Arrays.asList(0, 1 % numGPU, 2 % numGPU, 3 % numGPU, 4 % numGPU, 5 % numGPU, 6 % numGPU, 7 % numGPU, 8 % numGPU, 9 % numGPU));
            assertEquals(10, context.getStreamManager().getNumberOfStreams());
        }
    }

    @Test
    public void minTransferWithDepTest() throws UnsupportedTypeException {
        int[] gpus = {4, 8};
        for (int numGPU : gpus) {
            AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                    .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                    .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                    .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                    .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                    .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
            executeMockComputationAndValidate(joinPipelineMockComputation(context),
                    Arrays.asList(0, 1, 2, 3, 0, 2, 3));
            assertEquals(4, context.getStreamManager().getNumberOfStreams());
        }
    }

    @Test
    public void minTransferWithDepMultiGPUTest() throws UnsupportedTypeException {
        int[] gpus = {4, 8};
        for (int numGPU : gpus) {
            AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                    .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                    .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT)
                    .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                    .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                    .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
            executeMockComputationAndValidate(joinPipelineMockComputation(context),
                    Arrays.asList(0, 1, 2, 3, 0, 0, 3));
            assertEquals(4, context.getStreamManager().getNumberOfStreams());
        }
    }

    @Test
    public void minTransferWithDep2MultiGPUTest() throws UnsupportedTypeException {
        int[] gpus = {4, 8};
        for (int numGPU : gpus) {
            AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                    .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                    .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT)
                    .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                    .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                    .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
            // Computation 5/7 is scheduled on 0 because that's the ideal device chosen by MULTIGPU_EARLY_DISJOINT
            // (device 0 and 1 have the same amount of data), even though device 1 would have a suitable parent.
            // This also creates a new stream.
            // Computation 7/7 is scheduled on 0 because 0 has A,B while device 1,2,3 have only one array each;
            executeMockComputationAndValidate(joinPipeline2MockComputation(context),
                    Arrays.asList(0, 1, 2, 3, 0, 3, 0));
            assertEquals(5, context.getStreamManager().getNumberOfStreams());
        }
    }

    @Test
    public void minTransferWithDep3MultiGPUTest() throws UnsupportedTypeException {
        int[] gpus = {4, 8};
        for (int numGPU : gpus) {
            AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                    .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                    .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT)
                    .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                    .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                    .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
            // Computation 7/7 is scheduled on 0 because all devices have 1 array each, but GPU0 comes first
            executeMockComputationAndValidate(joinPipeline3MockComputation(context),
                    Arrays.asList(0, 1, 2, 3, 0, 3, 0));
            assertEquals(5, context.getStreamManager().getNumberOfStreams());
        }
    }

    @Test
    public void minTransferWithDep4MultiGPUTest() throws UnsupportedTypeException {
        int[] gpus = {4, 8};
        for (int numGPU : gpus) {
            AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                    .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                    .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT)
                    .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                    .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                    .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
            // Computation 7/7 is scheduled on 3 because GPU3 has both A and C
            executeMockComputationAndValidate(joinPipeline4MockComputation(context),
                    Arrays.asList(0, 1, 2, 3, 0, 3, 3));
            assertEquals(5, context.getStreamManager().getNumberOfStreams());
        }
    }


    @Test
    public void minTransferWithThreeGPUImageTest() throws UnsupportedTypeException {
        int numGPU = 3;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(ComplexExecutionDAGMockTest.imageMockComputation(context),
                Arrays.asList(
                        0, 0, 0,
                        0, 0,
                        0, 0,
                        0, 0, 0, 0, 0));
        assertEquals(IMAGE_NUM_STREAMS, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void minTransferWithFourGPUHitsTest() throws UnsupportedTypeException {
        int numGPU = 4;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        // After the first iteration, GPU0 has a3 up to date, GPU1 has a4. So GPU0 is chosen as it comes first,
        // and the scheduling collapses to GPU0;
        executeMockComputationAndValidate(ComplexExecutionDAGMockTest.hitsMockComputation(context),
                Arrays.asList(0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0));
        assertEquals(3, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void minTransferManyKernelsWithFourGPUTest() throws UnsupportedTypeException {
        int numGPU = 4;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        // The last 4 computations are scheduled on GPU0 as all devices contain just 1 required array and GPU0 is first;
        executeMockComputationAndValidate(manyKernelsMockComputation(context),
                Arrays.asList(0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 0, 0, 0, 0));
        assertEquals(7, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void minTransferForkJoinWithTwoGPUTest() throws UnsupportedTypeException {
        int numGPU = 2;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(forkJoinMockComputation(context),
                Arrays.asList(0, 1, 0, 0, 0));
        assertEquals(3, context.getStreamManager().getNumberOfStreams());
    }
}
