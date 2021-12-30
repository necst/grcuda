package com.nvidia.grcuda.runtime.stream.policy;

public enum DeviceSelectionPolicyEnum {
    SINGLE_GPU("single-gpu"),
    STREAM_AWARE("stream-aware"),
    DATA_LOCALITY("data-locality"),
    DATA_LOCALITY_NEW("data-locality-new"),
    TRANSFER_TIME_MIN("best-transfer-time-min"),
    TRANSFER_TIME_MAX("best-transfer-time-max");

    private final String name;

    DeviceSelectionPolicyEnum(String name) {
        this.name = name;
    }

    public final String getName() {
        return name;
    }
}