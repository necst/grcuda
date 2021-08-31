package com.nvidia.grcuda.gpu.stream;

public enum RetrieveParentStreamPolicyEnum {
<<<<<<< HEAD
    SAME_AS_PARENT("same-as-parent"),
=======
    DEFAULT("default"),
>>>>>>> improved testing interface to test all input options combinations at once
    DISJOINT("disjoint");

    private final String name;

    RetrieveParentStreamPolicyEnum(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
