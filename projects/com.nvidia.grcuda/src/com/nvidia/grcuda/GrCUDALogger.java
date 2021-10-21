package com.nvidia.grcuda;

import com.oracle.truffle.api.TruffleLogger;

public class GrCUDALogger {

    public static final String MAIN_LOGGER = "com.nvidia.grcuda";

    public static final String PARSER_LOGGER = "com.nvidia.grcuda.parser";

    public static TruffleLogger getLogger(String name) {
        return TruffleLogger.getLogger(GrCUDALanguage.ID, name);
    }
}
