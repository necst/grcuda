package com.nvidia.grcuda.runtime;

import java.util.Collection;
import com.nvidia.grcuda.runtime.stream.CUDAStream;

public class GrCUDADevicesManager {

    private final CUDARuntime runtime;
    private final DeviceList deviceList;

    /**
     * Initialize the GrCUDADevicesManager, creating a DeviceList that tracks all available GPUs.
     * @param runtime reference to the CUDA runtime
     */
    public GrCUDADevicesManager(CUDARuntime runtime) {
        this(runtime, new DeviceList(runtime));
    }

    /**
     * Initialize the GrCUDADevicesManager, using an existing DeviceList that tracks all available GPUs;
     * @param runtime reference to the CUDA runtime
     * @param deviceList list of available devices
     */
    public GrCUDADevicesManager(CUDARuntime runtime, DeviceList deviceList) {
        this.runtime = runtime;
        this.deviceList = deviceList;
    }
//
//    /**
//     * Check if there are available streams on the selected device
//     *
//     * @param deviceId
//     * @return boolean
//     */
//    public boolean availableStreams(int deviceId){
//        return deviceList.getDevice(deviceId).numFreeStreams() != 0;
//    }
//
//    /**
//     * Check if there are available streams on the selected device
//     *
//     * @return boolean
//     */
//    public boolean availableStreams(){
//        int deviceId = runtime.getCurrentGPU();
//        return deviceList.getDevice(deviceId).numFreeStreams() != 0;
//    }
//
//    /**
//     * update the list of free streams with the new stream with respect to the device with which it is associated
//     *
//     * @param stream
//     */
//    public void updateStreams(CUDAStream stream){
//        Device device = deviceList.getDevice(stream.getStreamDeviceId());
//        device.updateStreams(stream);
//    }
//
//    /**
//     * update the list of free streams with the new collection of streams with respect to the device with which it is associated
//     * @param streams
//     */
//    public void updateStreams(Collection<CUDAStream> streams){
//        for (CUDAStream stream: streams){
//            updateStreams(stream);
//        }
//    }
//
//    /**
//     * Retrive a free Stream from the device selected
//     * @param deviceId
//     * @return {@link CUDAStream}
//     */
//    public CUDAStream retriveStream(int deviceId){
//        Device device = deviceList.getDevice(deviceId);
//        return device.getFreeStream();
//
//    }
//
//    /**
//     * Find the device with the lowest number of Stream on it and returns it
//     * @return deviceId
//     */
//    public int deviceWithLessActiveStream(){
//        int min = deviceList.getDevice(0).numActiveStream();
//        int deviceId = 0;
//        for(int i = 0; i<numberOfGPUs; i++){
//            if(deviceList.getDevice(i).numActiveStream() < min){
//                min = deviceList.getDevice(i).numActiveStream();
//                deviceId = i;
//            }
//        }
//        return deviceId;
//    }
//
//    public int[] devicesActiveStreams(){
//        int[] activeStream = new int[deviceList.size()];
//        for(int i = 0; i<deviceList.size(); i++){
//            activeStream[i] = deviceList.getDevice(i).numActiveStream();
//        }
//        return activeStream;
//    }
//
//    public void addStreamCount(int deviceId){
//        deviceList.getDevice(deviceId).increaseStreamCount();
//    }

    public int getCurrentGPUId(){
        return this.runtime.getCurrentGPU();
    }

    public Device getCurrentGPU(){
        return this.getDevice(this.runtime.getCurrentGPU());
    }

    public int getNumberOfGPUsToUse(){
        return this.runtime.getNumberOfGPUsToUse();
    }

    public void setCurrentGPU(int deviceId) {
        this.runtime.setCurrentGPU(deviceId);
    }

    public DeviceList getDeviceList() {
        return deviceList;
    }

    public Device getDevice(int deviceId) {
        return deviceList.getDevice(deviceId);
    }

    /**
     * Cleanup and deallocate the streams managed by this manager;
     */
    public void cleanup() {
        this.deviceList.cleanup();
    }
}