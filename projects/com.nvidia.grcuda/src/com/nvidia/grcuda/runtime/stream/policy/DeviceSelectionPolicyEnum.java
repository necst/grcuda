package com.nvidia.grcuda.runtime.stream.policy;

public enum DeviceSelectionPolicyEnum {
    SINGLE_GPU("single-gpu"),
    ROUND_ROBIN("round-robin"),
    STREAM_AWARE("stream-aware"),
    MIN_TRANSFER_SIZE("min-transfer-size"),
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