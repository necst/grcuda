package com.nvidia.grcuda.test.runtime.stream;

import com.nvidia.grcuda.GrCUDAOptionMap;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import com.nvidia.grcuda.test.util.mock.AsyncGrCUDAExecutionContextMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAExecutionContextMockBuilder;
import com.nvidia.grcuda.test.util.mock.OptionValuesMockBuilder;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.bsMultiGPUMockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.executeMockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.executeMockComputationAndValidate;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.mlMultiGPUMockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.vecMultiGPUMockComputation;
import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

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

    private AsyncGrCUDAExecutionContextMock buildContext(boolean debug) {
        AsyncGrCUDAExecutionContextMock context = new AsyncGrCUDAExecutionContextMock(
                this.retrieveNewStreamPolicy,
                this.retrieveParentStreamPolicy,
                this.deviceSelectionPolicy,
                true, this.numberOfGPUs, this.numberOfGPUs,
                new GrCUDAOptionMap(new OptionValuesMockBuilder()
                        .add(GrCUDAOptions.DependencyPolicy, DependencyPolicyEnum.WITH_CONST.toString())
                        .add(GrCUDAOptions.InputPrefetch, false)
                        .add(GrCUDAOptions.BandwidthMatrix, System.getenv("GRCUDA_HOME") + File.separatorChar +
                                "projects" + File.separatorChar + "resources" + File.separatorChar +
                                "connection_graph" + File.separatorChar + "datasets" + File.separatorChar + "connection_graph_test_8_v100.csv").build()));
        if (debug) {
            System.out.println(this);
        }
        return context;
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
    public void vecMultiGPUMockTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContextMock context = buildContext(true);
        List<Integer> scheduling = new ArrayList<>();
        for (int i = 0; i < 2 * GrCUDAComputationsMock.partitionsVec / this.numberOfGPUs; i++) {
            for (int j = 0; j < this.numberOfGPUs / 2; j++) {
                scheduling.add(j * 2);
                scheduling.add(1 + j * 2);
                scheduling.add(j * 2);
            }
        }
        for (int i = 0; i < GrCUDAComputationsMock.partitionsVec; i++) {
            scheduling.add(0);  // Sync computations are associated to device 0, even if they are run by the CPU;
        }
        executeMockComputationAndValidate(vecMultiGPUMockComputation(context), scheduling,true);
        assertEquals(2 * GrCUDAComputationsMock.partitionsVec, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void bsMultiGPUMockTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContextMock context = buildContext(true);
        List<Integer> scheduling = new ArrayList<>();
        for (int i = 0; i < GrCUDAComputationsMock.partitionsBs; i++) {
            scheduling.add(i % this.numberOfGPUs);
        }
        for (int i = 0; i < GrCUDAComputationsMock.partitionsBs; i++) {
            scheduling.add(0);  // Sync computations are associated to device 0, even if they are run by the CPU;
        }
        executeMockComputationAndValidate(bsMultiGPUMockComputation(context), scheduling,true);
        assertEquals(GrCUDAComputationsMock.partitionsBs, context.getStreamManager().getNumberOfStreams());
    }

    @Test
    public void mlMultiGPUMockTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContextMock context = buildContext(true);
        assumeTrue(this.retrieveParentStreamPolicy == RetrieveParentStreamPolicyEnum.MULTIGPU_DISJOINT);
        List<Integer> scheduling = new ArrayList<>();
        // RR1;
        for (int i = 0; i < GrCUDAComputationsMock.partitionsMl; i++) {
            scheduling.add(i % this.numberOfGPUs);
        }
        // RR11;
        for (int i = 0; i < GrCUDAComputationsMock.partitionsMl; i++) {
            scheduling.add(0);
        }
        // RR12, RR2;
        for (int i = 0; i < GrCUDAComputationsMock.partitionsMl; i++) {
            scheduling.add(i % this.numberOfGPUs);
            scheduling.add(i % this.numberOfGPUs);
        }
        // RR3, RRSF;
        scheduling.add(0);
        scheduling.add(0);
        // NB1;
        for (int i = 0; i < GrCUDAComputationsMock.partitionsMl; i++) {
            scheduling.add(i % this.numberOfGPUs);
        }
        // NB2, NB3. NB4, NBSF, AMAX, sync;
        scheduling.add(0);
        scheduling.add(0);
        scheduling.add(0);
        scheduling.add(0);
        scheduling.add(0);
        scheduling.add(0);
        executeMockComputationAndValidate(mlMultiGPUMockComputation(context), scheduling, true);
        assertEquals(3 * GrCUDAComputationsMock.partitionsMl, context.getStreamManager().getNumberOfStreams());
    }
}
