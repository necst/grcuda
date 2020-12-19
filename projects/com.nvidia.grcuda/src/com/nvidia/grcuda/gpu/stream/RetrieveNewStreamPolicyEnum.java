package com.nvidia.grcuda.gpu.stream;

public enum RetrieveNewStreamPolicyEnum {
    FIFO("fifo"),
    ALWAYS_NEW("always_new"),
    MULTI_FIFO("multi_fifo"),
    MULTI_ALWAYS_NEW("multi_always_new");
    private final String name;

    RetrieveNewStreamPolicyEnum(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
