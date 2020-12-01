package com.nvidia.grcuda;

public enum OOBProtectionPolicyEnum {
    NO_PROTECTION("no_protection"),
    PREVENT("prevent"),
    TRACK("track"),
    PREVENT_AND_TRACK("prevent_and_track");

    private final String name;

    OOBProtectionPolicyEnum(String name) {
        this.name = name;
    }

    public final String getName() {
        return name;
    }
}
