package com.nvidia.grcuda.gpu.stream;

public enum RetrieveParentStreamPolicyEnum {
    DEFAULT("default"),
    DISJOINT("disjoint"),
    DATA_AWARE("data_aware"),
    DISJOINT_DATA_AWARE("disjoint_data_aware"),
    STREAM_AWARE("stream_aware");

    private final String name;

    RetrieveParentStreamPolicyEnum(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
