package com.nvidia.grcuda.runtime.stream;

import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.stream.CUDAStream;

/**
 * Abstract class to manage async streams for supported libraries
 */

abstract public class LibrarySetStreamFunction {

    protected final Function setStreamFunctionNFI;

    protected LibrarySetStreamFunction(Function setStreamFunctionNFI) {
        this.setStreamFunctionNFI = setStreamFunctionNFI;
    }

    public abstract void setStream(CUDAStream stream);

}
