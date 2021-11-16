package com.nvidia.grcuda.runtime.stream;

import com.nvidia.grcuda.cudalibraries.tensorrt.TensorRTRegistry;
import com.nvidia.grcuda.functions.Function;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectInt;

public class TensorRTSetStreamFunction extends LibrarySetStreamFunction {

    private int context;
    private int engine;
    private int input_index, output_index;
    private CUDAStream stream;
    private String input_name, output_name;
    private final int buffers[];
    private final Function createExecutionContextFunction;  //TensorRTRegistry.TensorRTFunctionNFI.TRT_CREATE_EXECUTION_CONTEXT
    private final Function getBindingIndexesFunction;       //TensorRTRegistry.TensorRTFunctionNFI.TRT_GET_BINDING_INDEX

    public TensorRTSetStreamFunction (Function enqueueV2FunctionNFI, Function createExecutionContextFunction, Function getBindingIndexesFunction) {
        super(enqueueV2FunctionNFI);
        this.createExecutionContextFunction = createExecutionContextFunction;
        this.getBindingIndexesFunction = getBindingIndexesFunction;
        buffers = new int[2];
    }

    public void setStream(CUDAStream stream){
        this.stream = stream;
        Object[] setStreamArgs = {context, engine, buffers, stream.getRawPointer(), null};
        try {   //Possible fail in this function because of unmatching parameters
            INTEROP.execute(setStreamFunctionNFI,setStreamArgs);
        } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("failed to execute TensorRT stream");
            e.printStackTrace();
        }
    }

    public void createExecutionContext(int engine) {
        this.engine = engine;
        Object[] createExecutionContext = {engine};
        try {
            this.context = expectInt(INTEROP.execute(createExecutionContextFunction, createExecutionContext));
        } catch(ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("failed to create execution context");
            e.printStackTrace();
        }
    }

    public void getBindingIndexes(int input_buffer, String input_name) {
        this.input_name = input_name;
        Object[] getBindingIndexesArgsInput = {engine, input_name};
        try {
            this.input_index = expectInt(INTEROP.execute(getBindingIndexesFunction, getBindingIndexesArgsInput));
        } catch(ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("Failed to bind input Indexes");
            e.printStackTrace();
        }
        buffers[input_index] = input_buffer;
    }

    public void getBindingIndexesInput(int output_buffer, String output_name) {
        this.output_name = output_name;
        Object[] getBindingIndexesArgsOutput = {engine, output_name};
        try {
            this.output_index = expectInt(INTEROP.execute(getBindingIndexesFunction, getBindingIndexesArgsOutput));
        } catch(ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("Failed to bind output Indexes");
            e.printStackTrace();
        }
        buffers[output_index] = output_buffer;
    }
} 