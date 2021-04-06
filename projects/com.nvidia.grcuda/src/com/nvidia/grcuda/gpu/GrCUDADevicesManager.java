package com.nvidia.grcuda.gpu;

import java.util.ArrayList;
import java.util.Collection;
import java.util.stream.IntStream;

import com.nvidia.grcuda.gpu.stream.CUDAStream;

public class GrCUDADevicesManager {

    private final CUDARuntime runtime;
    private final ArrayList<Device> devices = new ArrayList<>();
    private final Integer numberOfGPUs;
    private Integer currentDeviceId;

    public GrCUDADevicesManager(CUDARuntime runtime, int device_number){
        this.runtime = runtime;
        
        int deviceNumberInSystem = runtime.cudaGetDeviceCount();
        if(device_number < deviceNumberInSystem){
            this.numberOfGPUs = device_number;
        }else{
            this.numberOfGPUs = deviceNumberInSystem;
        }

        System.out.println("number of GPUs in GrCUDADeviceManager "+this.numberOfGPUs.toString());

        this.currentDeviceId = runtime.cudaGetDevice();
        initDevices();

    }



    private void initDevices(){
        for(int i = 0; i<numberOfGPUs;i++) {
            //set the device in order to determine the deviceId
            runtime.cudaSetDevice(i);
            devices.add(new Device(i, runtime));
        }
        runtime.cudaSetDevice(0);
    }


    /**
     * Check if there are available streams on the selected device
     * 
     * @param deviceId
     * @return boolean
     */
    public boolean availableStreams(int deviceId){
        if(devices.get(deviceId).numFreeStreams()!=0){
            return true;
        }else{
            return false;
        }
    }

    /**
     * Check if there are available streams on the selected device
     * 
     * @return boolean
     */
    public boolean availableStreams(){
        int deviceId = runtime.cudaGetDevice();
        if(devices.get(deviceId).numFreeStreams()!=0){
            return true;
        }else{
            return false;
        }
    }

    /**
     * update the list of free streams with the new stream with respect to the device with which it is associated
     * 
     * @param stream
     */
    public void updateStreams(CUDAStream stream){
        Device device = devices.get(stream.getStreamDeviceId());
        device.updateStreams(stream);
    }

    /**
     * update the list of free streams with the new collection of streams with respect to the device with which it is associated
     * @param streams
     */
    public void updateStreams(Collection<CUDAStream> streams){
        for (CUDAStream stream: streams){
            updateStreams(stream);
        }
    }

    /**
     * Retrive a free Stream from the device selected
     * @param deviceId
     * @return {@link CUDAStream}
     */
    public CUDAStream retriveStream(int deviceId){
        Device device = devices.get(deviceId);
        return device.getFreeStream();

    }

    /**
     * Find the device with the lowest number of Stream on it and returns it
     * @return deviceId 
     */
    public int deviceWithLessActiveStream(){
        int min = devices.get(0).numActiveStream();
        int deviceId = 0;
        for(int i = 0; i<numberOfGPUs; i++){
            if(devices.get(i).numActiveStream() < min){
                min = devices.get(i).numActiveStream();
                deviceId = i;
            }
        }
        return deviceId;
    }

    public void addStreamCount(int deviceId){
        devices.get(deviceId).increaseStreamCount();
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
