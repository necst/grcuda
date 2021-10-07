package com.nvidia.grcuda.runtime.computation;

import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.stream.CUDAStream;

import java.util.stream.Stream;

abstract public class LibrarySetStreamFunction extends Function {

    protected final Function setStreamFunctionNFI;

    protected LibrarySetStreamFunction(String name, Function setStreamFunctionNFI) {
        super(name);
        this.setStreamFunctionNFI=setStreamFunctionNFI;
    }

    public abstract void setStream(CUDAStream stream);

}
