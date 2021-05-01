package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.GrCUDADevicesManager;

public class GrCUDADeviceManagerMock extends GrCUDADevicesManager {
    public GrCUDADeviceManagerMock(int device_number) {
        super(null, device_number, 0, new DeviceListMock(device_number));
    }


}
