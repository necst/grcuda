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
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.vecMultiGPUMockComputation;
import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class MultiGPUComplexDAGMockTest {

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                {RetrieveNewStreamPolicyEnum.ALWAYS_NEW},
                {RetrieveParentStreamPolicyEnum.SAME_AS_PARENT, RetrieveParentStreamPolicyEnum.DISJOINT, RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT, RetrieveParentStreamPolicyEnum.MULTIGPU_DISJOINT},
                {DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE, DeviceSelectionPolicyEnum.MINMAX_TRANSFER_TIME, DeviceSelectionPolicyEnum.MINMIN_TRANSFER_TIME},
                {2, 4, 8}
        }));
    }

    private final RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy;
    private final RetrieveParentStreamPolicyEnum retrieveParentStreamPolicy;
    private final DeviceSelectionPolicyEnum deviceSelectionPolicy;
    private final int numberOfGPUs;

    public MultiGPUComplexDAGMockTest(
            RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy,
            RetrieveParentStreamPolicyEnum retrieveParentStreamPolicy,
            DeviceSelectionPolicyEnum deviceSelectionPolicy,
            int numberOfGPUs) {
        this.retrieveNewStreamPolicy = retrieveNewStreamPolicy;
        this.retrieveParentStreamPolicy = retrieveParentStreamPolicy;
        this.deviceSelectionPolicy = deviceSelectionPolicy;
        this.numberOfGPUs = numberOfGPUs;
    }

    @Override
    public String toString() {
        return "options{" +
                "new-stream=" + retrieveNewStreamPolicy +
                ", parent-stream=" + retrieveParentStreamPolicy +
                ", device-selection=" + deviceSelectionPolicy +
                ", gpu-num=" + numberOfGPUs +
                '}';
    }

    @Test
    public void minTransferWithThreeGPUImageTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(this.retrieveParentStreamPolicy)
                .setDeviceSelectionPolicy(this.deviceSelectionPolicy)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(this.numberOfGPUs).setNumberOfAvailableGPUs(this.numberOfGPUs).build();
        System.out.println(this);
        executeMockComputation(vecMultiGPUMockComputation(context), true);
        assertEquals(4, context.getStreamManager().getNumberOfStreams());
    }
}
