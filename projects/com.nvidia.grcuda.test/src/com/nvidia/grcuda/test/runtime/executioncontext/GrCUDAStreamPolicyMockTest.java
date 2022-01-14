package com.nvidia.grcuda.test.runtime.executioncontext;

import com.nvidia.grcuda.GrCUDAOptionMap;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.GrCUDAStreamPolicy;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.util.mock.AsyncGrCUDAExecutionContextMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAStreamPolicyMock;
import com.nvidia.grcuda.test.util.mock.OptionValuesMockBuilder;
import org.graalvm.compiler.hotspot.stubs.OutOfBoundsExceptionStub;
import org.junit.Test;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;

import static org.junit.Assert.assertEquals;

public class GrCUDAStreamPolicyMockTest {

    private static AsyncGrCUDAExecutionContextMock createContext(int numberOfGPUs, DeviceSelectionPolicyEnum deviceSelectionPolicy) {
        return new AsyncGrCUDAExecutionContextMock(
                RetrieveNewStreamPolicyEnum.ALWAYS_NEW,
                RetrieveParentStreamPolicyEnum.DISJOINT,
                deviceSelectionPolicy,
                true, numberOfGPUs, numberOfGPUs,
                new GrCUDAOptionMap(new OptionValuesMockBuilder()
                        .add(GrCUDAOptions.DependencyPolicy, DependencyPolicyEnum.WITH_CONST.toString())
                        .add(GrCUDAOptions.InputPrefetch, false)
                        .add(GrCUDAOptions.BandwidthMatrix, System.getenv("GRCUDA_HOME") + File.separatorChar +
                                "projects" + File.separatorChar + "resources" + File.separatorChar +
                                "connection_graph" + File.separatorChar + "datasets" + File.separatorChar + "connection_graph_test.csv").build())
        );
    }

    @Test
    public void createBandwidthMatrixTest() {
        AsyncGrCUDAExecutionContextMock context = createContext(2, DeviceSelectionPolicyEnum.MINMAX_TRANSFER_TIME);
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

    @Test
    public void bandwidthComputationMinMaxTest() {
        AsyncGrCUDAExecutionContextMock context = createContext(2, DeviceSelectionPolicyEnum.MINMAX_TRANSFER_TIME);
        GrCUDAStreamPolicy.TransferTimeDeviceSelectionPolicy deviceSelectionPolicy = (GrCUDAStreamPolicy.TransferTimeDeviceSelectionPolicy) ((GrCUDAStreamPolicyMock) context.getStreamManager().getStreamPolicy()).getDeviceSelectionPolicy();
        // If data is updated on the target device, we have infinite bandwidth (regardless of what's on the matrix diagonal);
        double b = deviceSelectionPolicy.computeBandwidth(0, new HashSet<>(Arrays.asList(0, 1, CPUDevice.CPU_DEVICE_ID)));
        assertEquals(Double.POSITIVE_INFINITY, b, 1e-6);
        // If the data is updated on another device, take the worst bandwidth;
        b = deviceSelectionPolicy.computeBandwidth(0, new HashSet<>(Arrays.asList(1, CPUDevice.CPU_DEVICE_ID)));
        assertEquals(10, b, 1e-6);
    }

    @Test
    public void bandwidthComputationMinMinTest() {
        AsyncGrCUDAExecutionContextMock context = createContext(2, DeviceSelectionPolicyEnum.MINMIN_TRANSFER_TIME);
        GrCUDAStreamPolicy.TransferTimeDeviceSelectionPolicy deviceSelectionPolicy = (GrCUDAStreamPolicy.TransferTimeDeviceSelectionPolicy) ((GrCUDAStreamPolicyMock) context.getStreamManager().getStreamPolicy()).getDeviceSelectionPolicy();
        // If data is updated on the target device, we have infinite bandwidth (regardless of what's on the matrix diagonal);
        double b = deviceSelectionPolicy.computeBandwidth(0, new HashSet<>(Arrays.asList(0, 1, CPUDevice.CPU_DEVICE_ID)));
        assertEquals(Double.POSITIVE_INFINITY, b, 1e-6);
        // If the data is updated on another device, take the worst bandwidth;
        b = deviceSelectionPolicy.computeBandwidth(0, new HashSet<>(Arrays.asList(1, CPUDevice.CPU_DEVICE_ID)));
        assertEquals(45, b, 1e-6);
    }

    @Test(expected = IllegalStateException.class)
    public void bandwidthComputationWithNoUpdatedLocationTest() {
        AsyncGrCUDAExecutionContextMock context = createContext(2, DeviceSelectionPolicyEnum.MINMAX_TRANSFER_TIME);
        GrCUDAStreamPolicy.TransferTimeDeviceSelectionPolicy deviceSelectionPolicy = (GrCUDAStreamPolicy.TransferTimeDeviceSelectionPolicy) ((GrCUDAStreamPolicyMock) context.getStreamManager().getStreamPolicy()).getDeviceSelectionPolicy();
        // If the data is not available on any device, give an error;
        double b = deviceSelectionPolicy.computeBandwidth(0, new HashSet<>());
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void bandwidthComputationOutOfBoundsLocationTest() {
        AsyncGrCUDAExecutionContextMock context = createContext(2, DeviceSelectionPolicyEnum.MINMAX_TRANSFER_TIME);
        GrCUDAStreamPolicy.TransferTimeDeviceSelectionPolicy deviceSelectionPolicy = (GrCUDAStreamPolicy.TransferTimeDeviceSelectionPolicy) ((GrCUDAStreamPolicyMock) context.getStreamManager().getStreamPolicy()).getDeviceSelectionPolicy();
        // If the data is not available on any device, give an error;
        double b = deviceSelectionPolicy.computeBandwidth(10, new HashSet<>(Collections.singletonList(1)));
    }
}
