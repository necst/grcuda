package com.nvidia.grcuda.gpu.computation.prefetch;

public enum PrefetcherEnum {
    NONE("none"),
    DEFAULT("default"),
    SYNC("sync");

    private final String name;

    PrefetcherEnum(String name){ this.name=name; }

    public final String getName() {
        return name;
    }
}
