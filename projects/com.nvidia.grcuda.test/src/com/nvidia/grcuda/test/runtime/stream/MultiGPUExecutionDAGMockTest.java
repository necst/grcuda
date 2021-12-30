package com.nvidia.grcuda.test.runtime.stream;

import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import com.nvidia.grcuda.test.util.mock.GrCUDAExecutionContextMockBuilder;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
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
                System.out.println("wrong GPU allocation for " + c + "; expected=" + expected + "; actual=" + actual);
            }
            assertEquals(expected, actual);
        }
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
            // If using ALWAYS_NEW, streams are always set to busy and are never free (because they cannot be reused anyway);
            assertEquals(this.retrieveNewStreamPolicy == RetrieveNewStreamPolicyEnum.REUSE ? IMAGE_NUM_STREAMS : 0, context.getStreamManager().getDevice(0).getNumberOfFreeStreams());
            assertEquals(this.retrieveNewStreamPolicy == RetrieveNewStreamPolicyEnum.REUSE ? 0 : IMAGE_NUM_STREAMS, context.getStreamManager().getDevice(0).getNumberOfBusyStreams());
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
        assertEquals(this.retrieveNewStreamPolicy == RetrieveNewStreamPolicyEnum.REUSE ? IMAGE_NUM_STREAMS : 0, context.getStreamManager().getDevice(0).getNumberOfFreeStreams());
        assertEquals(this.retrieveNewStreamPolicy == RetrieveNewStreamPolicyEnum.REUSE ? 0 : IMAGE_NUM_STREAMS, context.getStreamManager().getDevice(0).getNumberOfBusyStreams());
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
            // If using ALWAYS_NEW, streams are always set to busy and are never free (because they cannot be reused anyway);
            assertEquals(this.retrieveNewStreamPolicy == RetrieveNewStreamPolicyEnum.REUSE ? HITS_NUM_STREAMS : 0, context.getStreamManager().getDevice(0).getNumberOfFreeStreams());
            assertEquals(this.retrieveNewStreamPolicy == RetrieveNewStreamPolicyEnum.REUSE ? 0 : HITS_NUM_STREAMS, context.getStreamManager().getDevice(0).getNumberOfBusyStreams());
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
        assertEquals(this.retrieveNewStreamPolicy == RetrieveNewStreamPolicyEnum.REUSE ? HITS_NUM_STREAMS : 0, context.getStreamManager().getDevice(0).getNumberOfFreeStreams());
        assertEquals(this.retrieveNewStreamPolicy == RetrieveNewStreamPolicyEnum.REUSE ? 0 : HITS_NUM_STREAMS, context.getStreamManager().getDevice(0).getNumberOfBusyStreams());
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
}
