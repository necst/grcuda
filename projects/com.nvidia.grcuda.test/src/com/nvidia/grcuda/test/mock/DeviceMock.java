package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.Device;

public class DeviceMock extends Device{
    public DeviceMock(int deviceId){
        super(deviceId, null, null);
    }
}
