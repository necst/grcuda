package com.nvidia.grcuda.runtime.computation;

import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

/**
 * Class of functions to manage streams in the CUBLAS library
 */

public class CUBLASSetStreamFunction extends LibrarySetStreamFunction {

    private final long handle;

    public CUBLASSetStreamFunction(String name, Function setStreamFunctionNFI, long handle) {
        super(name, setStreamFunctionNFI);
        this.handle = handle;
    }

    @Override
    public void setStream(CUDAStream stream) {
        Object[] cublasSetStreamArgs = {this.handle, stream.getRawPointer()};
        try {
            INTEROP.execute(this.setStreamFunctionNFI, cublasSetStreamArgs);
        } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("failed to set CUBLAS stream");
            e.printStackTrace();
        }
    }
}
