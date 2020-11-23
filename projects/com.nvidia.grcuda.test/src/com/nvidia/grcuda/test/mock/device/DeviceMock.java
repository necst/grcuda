package com.nvidia.grcuda.test.mock.device;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.Device;

public class DeviceMock extends Device {
    public DeviceMock(int deviceId, CUDARuntime runtime) {
        super(deviceId, runtime);
    }


}
