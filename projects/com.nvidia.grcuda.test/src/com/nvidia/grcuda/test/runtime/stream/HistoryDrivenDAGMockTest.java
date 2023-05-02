package com.nvidia.grcuda.test.runtime.stream;

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

import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.executeMockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.executeMockComputationAndValidate;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.forkJoinMockComputationWithTime;
import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class HistoryDrivenDAGMockTest {

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                {RetrieveNewStreamPolicyEnum.ALWAYS_NEW, RetrieveNewStreamPolicyEnum.REUSE}
        }));
    }

    private final RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy;

    public HistoryDrivenDAGMockTest(RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy) {
        this.retrieveNewStreamPolicy = retrieveNewStreamPolicy;
    }

    private final static int IMAGE_NUM_STREAMS = 4;
    private final static int HITS_NUM_STREAMS = 2;

    @Test
    public void HistoryDrivenForkJoinWithTwoGPUTest() throws UnsupportedTypeException {
        int numGPU = 2;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MINMAX_PARALLEL_HISTORY_DRIVEN)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(forkJoinMockComputationWithTime(context),
                Arrays.asList(0, 1, 1, 1, 1));
        assertEquals(3, context.getStreamManager().getNumberOfStreams());
    }
}
