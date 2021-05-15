package com.nvidia.grcuda.gpu.stream;

public enum RetrieveParentStreamPolicyEnum {
    DEFAULT("default"),
    DISJOINT("disjoint"),
    MORE_ARGUMENT("more_argument"),
    DISJOINT_MORE_ARGUMENT("disjoint_more_argument");

    private final String name;

    RetrieveParentStreamPolicyEnum(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
