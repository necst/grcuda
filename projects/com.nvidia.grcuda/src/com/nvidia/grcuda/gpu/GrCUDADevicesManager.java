package com.nvidia.grcuda.gpu;

import java.util.ArrayList;

public class GrCUDADevicesManager {
    private final CUDARuntime runtime;
    private final ArrayList<Device> devices = new ArrayList<>();
    private final Integer numberOfGPUs;
    private Integer currentDeviceId;
    public GrCUDADevicesManager(CUDARuntime runtime){
        this.runtime = runtime;
        this.numberOfGPUs = runtime.cudaGetDeviceCount();
        this.currentDeviceId = runtime.cudaGetDevice();
        initDevices();
    }

    private void initDevices(){
        for(int i = 0; i<numberOfGPUs;i++) {
            devices.add(new Device(i, runtime));
        }
    }

    public int getCurrentDeviceId(){
        return this.currentDeviceId;
    }

    public int getNumberOfGPUs(){
        return this.numberOfGPUs;
    }

    public void setDevice(int id){
        runtime.cudaSetDevice(id);
        this.currentDeviceId = id;
    }
}
