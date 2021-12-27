package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.GrCUDADevicesManager;

public class GrCUDADevicesManagerMock extends GrCUDADevicesManager {

    private int currentGPU = 0;
    final private int numberOfGPUsToUse;

    public GrCUDADevicesManagerMock(DeviceListMock deviceList, int numberOfGPUsToUse) {
        super(null, deviceList);
        this.numberOfGPUsToUse = numberOfGPUsToUse;
    }

    @Override
    public int getCurrentGPUId(){
        return currentGPU;
    }

    @Override
    public Device getCurrentGPU(){
        return this.getDevice(this.currentGPU);
    }

    @Override
    public int getNumberOfGPUsToUse(){
        return numberOfGPUsToUse;
    }

    @Override
    public void setCurrentGPU(int deviceId) {
        currentGPU = deviceId;
    }
}
