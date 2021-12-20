package com.nvidia.grcuda.runtime.stream;

import com.nvidia.grcuda.functions.Function;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectInt;

public class TensorRTSetStreamFunction extends LibrarySetStreamFunction {

    public TensorRTSetStreamFunction (Function enqueueV2FunctionNFI) {
        super(enqueueV2FunctionNFI);
    }

    public void setStream(CUDAStream stream){/* nothing to do */}
} 