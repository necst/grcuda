package com.nvidia.grcuda.runtime.stream;

import com.nvidia.grcuda.functions.Function;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectInt;

public class TensorRTSetStreamFunction extends LibrarySetStreamFunction {
    private final int BATCH_SIZE = 1;
    private int context;
    private int engine;
    private int inputIndex, outputIndex;
    private CUDAStream stream;
    private String inputName, outputName;
    private Long[] buffers;

    public TensorRTSetStreamFunction (Function enqueueV2FunctionNFI) {
        super(enqueueV2FunctionNFI);
    }

    public void setStream(CUDAStream stream){
        this.stream = stream;
        Object[] setStreamArgs = {BATCH_SIZE, buffers, stream.getRawPointer(), 0};
        try {
            INTEROP.execute(setStreamFunctionNFI,setStreamArgs);
        } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("Failed to execute TensorRT stream");
            e.printStackTrace();
        }
    }

    public void tensorRTSetBuffers(Long[] buffers) {
        this.buffers = buffers;
    }
} 