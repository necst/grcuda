package com.nvidia.grcuda.runtime;

/**
 * Abstract device representation, used to distinguish between CPU and GPU devices inside the GrCUDA scheduler.
 */
public class AbstractDevice {
    protected final int deviceId;

    public AbstractDevice(int deviceId) {
        this.deviceId = deviceId;
    }

    public int getDeviceId() {
        return deviceId;
    }

    @Override
    public String toString() {
        return "Device(id=" + deviceId + ")";
    }

    @Override
    public boolean equals(Object other) {
        if (other instanceof AbstractDevice) {
            AbstractDevice otherDevice = (AbstractDevice) other;
            return otherDevice.deviceId == deviceId;
        }
        return false;
    }

    @Override
    public int hashCode() {
        return deviceId;
    }
}
