package com.nvidia.grcuda.gpu.computation.dependency;

public enum DependencyPolicyEnum {
<<<<<<< HEAD
    NO_CONST("no-const"),
=======
    DEFAULT("default"),
>>>>>>> improved testing interface to test all input options combinations at once
    WITH_CONST("with-const");

    private final String name;

    DependencyPolicyEnum(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
