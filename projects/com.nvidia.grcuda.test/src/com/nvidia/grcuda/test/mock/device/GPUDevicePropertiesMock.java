package com.nvidia.grcuda.test.mock.device;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.GPUDeviceProperties;

public class GPUDevicePropertiesMock extends GPUDeviceProperties {

    public GPUDevicePropertiesMock(int deviceId, CUDARuntime runtime) {
        super(deviceId, runtime);
    }
}
