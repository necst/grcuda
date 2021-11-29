package com.nvidia.grcuda.runtime.stream;

import com.nvidia.grcuda.functions.Function;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectInt;

public class TensorRTSetStreamFunction extends LibrarySetStreamFunction {

    private int context;
    private int engine;
    private int inputIndex, outputIndex;
    private CUDAStream stream;
    private String inputName, outputName;
    private final int buffers[];
    private final Function createExecutionContextFunctionNFI;  //TensorRTRegistry.TensorRTFunctionNFI.TRT_CREATE_EXECUTION_CONTEXT
    private final Function getBindingIndexesFunctionNFI;       //TensorRTRegistry.TensorRTFunctionNFI.TRT_GET_BINDING_INDEX

    public TensorRTSetStreamFunction (Function enqueueV2FunctionNFI, Function createExecutionContextFunctionNFI, Function getBindingIndexesFunctionNFI  ) {
        super(enqueueV2FunctionNFI);
        this.createExecutionContextFunctionNFI = createExecutionContextFunctionNFI;
        this.getBindingIndexesFunctionNFI = getBindingIndexesFunctionNFI;
        buffers = new int[2];
    }

    public void setStream(CUDAStream stream){
        this.stream = stream;
        Object[] setStreamArgs = {context, engine, buffers, stream.getRawPointer(), null};
        try {   //Possible fail in this function because of unmatching parameters
            INTEROP.execute(setStreamFunctionNFI,setStreamArgs);
        } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("Failed to execute TensorRT stream");
            e.printStackTrace();
        }
    }

    public void createExecutionContext(int engine) {
        this.engine = engine;
        Object[] createExecutionContext = {engine};
        try {
            this.context = expectInt(INTEROP.execute(createExecutionContextFunctionNFI, createExecutionContext));
        } catch(ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("Failed to create execution context");
            e.printStackTrace();
        }
    }

    public void getBindingIndexes(int inputBuffer, String inputName) {
        this.inputName = inputName;
        Object[] getBindingIndexesArgsInput = {engine, inputName};
        try {
            this.inputIndex = expectInt(INTEROP.execute(getBindingIndexesFunctionNFI, getBindingIndexesArgsInput));
        } catch(ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("Failed to bind input Indexes");
            e.printStackTrace();
        }
        buffers[inputIndex] = inputBuffer;
    }

    public void getBindingIndexesInput(int outputBuffer, String outputName) {
        this.outputName = outputName;
        Object[] getBindingIndexesArgsOutput = {engine, outputName};
        try {
            this.outputIndex = expectInt(INTEROP.execute(getBindingIndexesFunctionNFI, getBindingIndexesArgsOutput));
        } catch(ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("Failed to bind output Indexes");
            e.printStackTrace();
        }
        buffers[outputIndex] = outputBuffer;
    }
} 