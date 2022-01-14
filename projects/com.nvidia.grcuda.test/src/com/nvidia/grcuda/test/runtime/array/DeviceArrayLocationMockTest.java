package com.nvidia.grcuda.test.runtime.array;

import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.test.util.mock.AsyncGrCUDAExecutionContextMock;
import com.nvidia.grcuda.test.util.mock.DeviceArrayMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAExecutionContextMockBuilder;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class DeviceArrayLocationMockTest {

    @Test
    public void testIfInitializedCorrectlyPrePascal() {
        AsyncGrCUDAExecutionContextMock context = new GrCUDAExecutionContextMockBuilder().setArchitecturePascalOrNewer(false).build();
        DeviceArray array1 = new DeviceArrayMock(context);
        DeviceArray array2 = new DeviceArrayMock(context);
        assertEquals(1, array1.getArrayUpToDateLocations().size());
        assertEquals(1, array2.getArrayUpToDateLocations().size());
        assertEquals(array1.getArrayUpToDateLocations(), array2.getArrayUpToDateLocations());
        assertTrue(array1.getArrayUpToDateLocations().contains(context.currentGPU));
    }

    @Test
    public void testIfInitializedCorrectlyPostPascal() {
        AsyncGrCUDAExecutionContextMock context = new GrCUDAExecutionContextMockBuilder().setArchitecturePascalOrNewer(true).build();
        DeviceArray array1 = new DeviceArrayMock(context);
        DeviceArray array2 = new DeviceArrayMock(context);
        assertEquals(1, array1.getArrayUpToDateLocations().size());
        assertEquals(1, array2.getArrayUpToDateLocations().size());
        assertEquals(array1.getArrayUpToDateLocations(), array2.getArrayUpToDateLocations());
        assertTrue(array1.isArrayUpdatedInLocation(CPUDevice.CPU_DEVICE_ID));
    }

    @Test
    public void testIfLocationAdded() {
        AsyncGrCUDAExecutionContextMock context = new GrCUDAExecutionContextMockBuilder().setArchitecturePascalOrNewer(true).build();
        DeviceArray array1 = new DeviceArrayMock(context);
        array1.addArrayUpToDateLocations(2);
        assertEquals(2, array1.getArrayUpToDateLocations().size());
        assertTrue(array1.isArrayUpdatedInLocation(2));
    }

    @Test
    public void testIfLocationReset() {
        AsyncGrCUDAExecutionContextMock context = new GrCUDAExecutionContextMockBuilder().setArchitecturePascalOrNewer(true).build();
        DeviceArray array1 = new DeviceArrayMock(context);
        array1.resetArrayUpToDateLocations(2);
        assertEquals(1, array1.getArrayUpToDateLocations().size());
        assertTrue(array1.isArrayUpdatedInLocation(2));
    }
}
