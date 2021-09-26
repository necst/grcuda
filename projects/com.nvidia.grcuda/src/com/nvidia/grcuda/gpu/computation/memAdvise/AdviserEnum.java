package com.nvidia.grcuda.gpu.computation.memAdvise;

public enum AdviserEnum {
    ADVISE_READ_MOSTLY("read_mostly"),
    ADVISE_PREFERRED_LOCATION("preferred"), // DefaultMemAdviser
    NONE("none");

    private final String name;

    AdviserEnum(String name){ this.name=name; }

    public final String getName() {
        return name;
    }
}
