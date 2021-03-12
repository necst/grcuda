package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.GrCUDADevicesManager;

public class GrCUDADeviceManagerMock extends GrCUDADevicesManager {
    public GrCUDADeviceManagerMock(CUDARuntime runtime) {
        super(runtime,2);
    }

}
