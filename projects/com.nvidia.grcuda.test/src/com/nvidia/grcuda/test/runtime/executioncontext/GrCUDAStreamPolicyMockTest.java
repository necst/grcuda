package com.nvidia.grcuda.test.runtime.executioncontext;

import com.nvidia.grcuda.GrCUDAOptionMap;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.GrCUDAStreamPolicy;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.util.mock.AsyncGrCUDAExecutionContextMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAStreamPolicyMock;
import com.nvidia.grcuda.test.util.mock.OptionValuesMockBuilder;
import org.junit.Test;

import java.io.File;

import static org.junit.Assert.assertEquals;

public class GrCUDAStreamPolicyMockTest {

    @Test
    public void createBandwidthMatrixTest() {
        AsyncGrCUDAExecutionContextMock context = new AsyncGrCUDAExecutionContextMock(
                RetrieveNewStreamPolicyEnum.ALWAYS_NEW,
                RetrieveParentStreamPolicyEnum.DISJOINT,
                DeviceSelectionPolicyEnum.MINMAX_TRANSFER_TIME,
                true, 2, 2,
                new GrCUDAOptionMap(new OptionValuesMockBuilder()
                        .add(GrCUDAOptions.DependencyPolicy, DependencyPolicyEnum.WITH_CONST.toString())
                        .add(GrCUDAOptions.InputPrefetch, false)
                        .add(GrCUDAOptions.BandwidthMatrix, System.getenv("GRCUDA_HOME") + File.separatorChar +
                                "projects" + File.separatorChar + "resources" + File.separatorChar +
                                "connection_graph" + File.separatorChar + "datasets" + File.separatorChar + "connection_graph_test.csv").build())
        );
        GrCUDAStreamPolicyMock streamPolicy = (GrCUDAStreamPolicyMock) context.getStreamManager().getStreamPolicy();
        double[][] bGold = {
                {30, 45, 10},
                {45, 60, 20},
                {10, 20, 0}
        };
        double[][] b = ((GrCUDAStreamPolicy.TransferTimeDeviceSelectionPolicy) streamPolicy.getDeviceSelectionPolicy()).getLinkBandwidth();
        for (int i = 0; i < b.length; i++) {
            for (int j = 0; j < b[i].length; j++) {
                assertEquals(bGold[i][j], b[i][j], 1e-6);
            }
        }
    }
}
