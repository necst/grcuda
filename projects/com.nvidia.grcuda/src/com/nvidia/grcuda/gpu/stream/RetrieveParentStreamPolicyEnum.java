package com.nvidia.grcuda.gpu.stream;

public enum RetrieveParentStreamPolicyEnum {
    DEFAULT("default"),
    DISJOINT("disjoint"),
    LESSTIME("lesstime"),
    MULTI_DISJOINT("multi_disjoint"),
    MULTI_DEFAULT("multi_default");

    private final String name;

    RetrieveParentStreamPolicyEnum(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
