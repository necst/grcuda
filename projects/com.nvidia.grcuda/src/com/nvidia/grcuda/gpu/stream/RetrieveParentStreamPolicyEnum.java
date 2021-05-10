package com.nvidia.grcuda.gpu.stream;

public enum RetrieveParentStreamPolicyEnum {
    DEFAULT("default"),
    DISJOINT("disjoint"),
    LESSTIME("lesstime");

    private final String name;

    RetrieveParentStreamPolicyEnum(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
