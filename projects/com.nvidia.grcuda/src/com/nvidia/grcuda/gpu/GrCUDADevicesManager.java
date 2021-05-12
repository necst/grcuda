package com.nvidia.grcuda.gpu;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.stream.IntStream;

import com.nvidia.grcuda.gpu.CUDARuntime.CUDADeviceAttribute;
import com.nvidia.grcuda.gpu.stream.CUDAStream;

public class GrCUDADevicesManager {

    private final CUDARuntime runtime;
    private final DeviceList deviceList;
    private final Integer numberOfGPUs;
    private Integer currentDeviceId;


    /**
     * Init the GrCUDADevicesManager class
     * @param runtime
     * @param device_number
     */
    public GrCUDADevicesManager(CUDARuntime runtime, int device_number){
        this(runtime, device_number,runtime.cudaGetDevice(), new DeviceList(device_number,runtime));
    }


    public GrCUDADevicesManager(CUDARuntime runtime, int device_number, int currentDeviceId, DeviceList deviceList){
        this.runtime = runtime;
        this.numberOfGPUs = device_number;
        this.currentDeviceId = currentDeviceId;
        this.deviceList = deviceList;
        for(int i = 0; i<this.numberOfGPUs; i++){
            System.out.println("device cudaDevAttrConcurrentManagedAccess "+runtime.cudaDeviceGetAttribute(CUDADeviceAttribute.CONCURRENT_MANAGED_ACCESS, i));
        }
        System.out.println("can access peer api 0 -> 1 :"+runtime.cudaDeviceCanAccessPeer(0, 1));
        System.out.println("can access peer api 1 -> 0 :"+runtime.cudaDeviceCanAccessPeer(1, 0));
        runtime.cudaDeviceEnablePeerAccess(0);
        System.out.println("can access peer api 0 -> 1 :"+runtime.cudaDeviceCanAccessPeer(0, 1));
        System.out.println("can access peer api 1 -> 0 :"+runtime.cudaDeviceCanAccessPeer(1, 0));
    }

    /**
     * check the number of GPUs in the system and check that device_number < cudaGetDeviceCount(). If the device_number is greater than the number of
     * devices in the system, the maximum number of devices is set.
     * @param device_number
     * @return
     */
    private int initNumOfDevices(int device_number){
        int deviceNumberInSystem = runtime.cudaGetDeviceCount();
        if(device_number < deviceNumberInSystem){
            return device_number;
        }else{
            return deviceNumberInSystem;
        }
    }

    /**
     * Check if there are available streams on the selected device
     * 
     * @param deviceId
     * @return boolean
     */
    public boolean availableStreams(int deviceId){
        if(deviceList.getDevice(deviceId).numFreeStreams()!=0){
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
        if(deviceList.getDevice(deviceId).numFreeStreams()!=0){
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
        Device device = deviceList.getDevice(stream.getStreamDeviceId());
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
        Device device = deviceList.getDevice(deviceId);
        return device.getFreeStream();

    }

    /**
     * Find the device with the lowest number of Stream on it and returns it
     * @return deviceId 
     */
    public int deviceWithLessActiveStream(){
        int min = deviceList.getDevice(0).numActiveStream();
        int deviceId = 0;
        for(int i = 0; i<numberOfGPUs; i++){
            if(deviceList.getDevice(i).numActiveStream() < min){
                min = deviceList.getDevice(i).numActiveStream();
                deviceId = i;
            }
        }
        return deviceId;
    }

    public int[] devicesActiveStreams(){
        int[] activeStream = new int[deviceList.size()];
        for(int i = 0; i<deviceList.size(); i++){
            activeStream[i] = deviceList.getDevice(i).numActiveStream();
        }
        return activeStream;
    }

    public void addStreamCount(int deviceId){
        deviceList.getDevice(deviceId).increaseStreamCount();
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
