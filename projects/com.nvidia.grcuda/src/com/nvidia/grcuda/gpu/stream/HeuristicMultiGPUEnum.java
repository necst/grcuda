package com.nvidia.grcuda.gpu.stream;

public enum HeuristicMultiGPUEnum {
    DATA_LOCALITY("data_locality"),
    TRANSFER_TIME_MIN("transfer_time_min"),
    TRANSFER_TIME_MAX("transfer_time_max");

    private final String name;

    ExecutionPolicyEnum(String name) {
        this.name = name;
    }

    public final String getName() {
        return name;
    }
}
