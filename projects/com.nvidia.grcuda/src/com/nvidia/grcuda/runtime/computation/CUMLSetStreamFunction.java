package com.nvidia.grcuda.runtime.computation;

import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CUMLSetStreamFunction extends LibrarySetStreamFunction {

    private long handle;

    public CUMLSetStreamFunction(String name, Function setStreamFunctionNFI, long handle) {
        super(name, setStreamFunctionNFI);
        this.handle=handle;
    }

    @Override
    public void setStream(CUDAStream stream) {
        Object[] cumlSetStreamArgs = {this.handle, stream.getRawPointer()}; // non devo richiedere streamID?
        try {
            INTEROP.execute(this.setStreamFunctionNFI, cumlSetStreamArgs);
        } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("failed to set CUML stream");
            e.printStackTrace();
        }
    }
}
