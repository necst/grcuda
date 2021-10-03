package com.nvidia.grcuda.gpu.stream;

public enum ChooseDeviceHeuristicEnum {
    DATA_LOCALITY("data_locality"),
    DATA_LOCALITY_NEW("data_locality_new"),
    TRANSFER_TIME_MIN("best_transfer_time_min"),
    TRANSFER_TIME_MAX("best_transfer_time_max");

    private final String name;

    ChooseDeviceHeuristicEnum(String name) {
        this.name = name;
    }

    public final String getName() {
        return name;
    }
}
