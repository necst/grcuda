package com.nvidia.grcuda.functions;

import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.SparseVector;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.ExportLibrary;

@ExportLibrary(InteropLibrary.class)
public class SparseVectorFunction extends Function {
    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    private final int NUM_ARGUMENTS = 4;

    public SparseVectorFunction(AbstractGrCUDAExecutionContext grCUDAExecutionContext) {
        super("SparseVector");
        this.grCUDAExecutionContext = grCUDAExecutionContext;
    }

    private boolean isDeviceArrayConstructor(Object[] arguments){
        return arguments[0] instanceof DeviceArray && arguments[1] instanceof DeviceArray;
    }


    @Override
    public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
        if (arguments.length != NUM_ARGUMENTS) {
            throw ArityException.create(NUM_ARGUMENTS, NUM_ARGUMENTS, arguments.length);
        }

        if(isDeviceArrayConstructor(arguments)){
            DeviceArray values = ((DeviceArray) arguments[0]);
            DeviceArray indices = ((DeviceArray) arguments[1]);
            long N = expectLong(arguments[2]);
            boolean isComplex = (Boolean) arguments[3];
            return createSparseVectorFromDeviceArrays(values, indices, N, isComplex);
        }

        throw UnsupportedTypeException.create(arguments, "Constructing SparseVectors is only allowed via `DeviceArray` for indices and values.");
    }

    private Object createSparseVectorFromDeviceArrays(DeviceArray values, DeviceArray indices, long N, boolean isComplex) throws UnsupportedTypeException {

        long numElements = isComplex ? values.getArraySize() / 2 : values.getArraySize();
        if(indices.getArraySize() != numElements){
            throw UnsupportedTypeException.create(new Object[]{values.getArraySize(), indices.getArraySize(), N, isComplex},"Indices and Values array must have the same size.");
        }

        return new SparseVector(this.grCUDAExecutionContext, values, indices, N, isComplex);
    }

}
