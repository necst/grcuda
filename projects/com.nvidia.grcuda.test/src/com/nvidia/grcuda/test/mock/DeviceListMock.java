package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.Device;
import com.nvidia.grcuda.gpu.DeviceList;


public class DeviceListMock extends DeviceList{
    private final Device[] devices;
    public DeviceListMock(int numDevices){ 
        super(numDevices);      
        devices = new Device[numDevices];
        for (int deviceOrdinal = 0; deviceOrdinal < numDevices; ++deviceOrdinal) {
            devices[deviceOrdinal] = new DeviceMock(deviceOrdinal);
        }
    }

    @Override
    public Device getDevice(int deviceOrdinal) {
        if ((deviceOrdinal < 0) || (deviceOrdinal >= devices.length)) {

            throw new IndexOutOfBoundsException();
        }
        return devices[deviceOrdinal];
    }

    @Override
    public int size() {
        return devices.length;
    }

}
