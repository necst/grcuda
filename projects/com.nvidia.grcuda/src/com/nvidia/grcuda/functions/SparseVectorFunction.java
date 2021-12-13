package com.nvidia.grcuda.functions;

import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.SparseVector;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.ExportLibrary;

@ExportLibrary(InteropLibrary.class)
public class SparseVectorFunction extends Function {
    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    private final int NUM_ARGUMENTS = 3;

    public SparseVectorFunction(AbstractGrCUDAExecutionContext grCUDAExecutionContext) {
        super("SparseVector");
        this.grCUDAExecutionContext = grCUDAExecutionContext;
    }

    private boolean isDeviceArrayConstructor(Object[] arguments){
        return arguments[0] instanceof DeviceArray && arguments[1] instanceof DeviceArray;
    }


    @Override
    public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
        if (arguments.length < NUM_ARGUMENTS) {
            throw ArityException.create(1, arguments.length);
        }

        if(isDeviceArrayConstructor(arguments)){
            DeviceArray indices = (DeviceArray) arguments[0];
            DeviceArray values = (DeviceArray) arguments[1];
            long N = expectLong(arguments[2]);
            return createSparseVectorFromDeviceArrays(indices, values, N);
        }

        throw UnsupportedTypeException.create(arguments, "Constructing SparseVectors is only allowed via `DeviceArray` for indices and values.");
    }

    private Object createSparseVectorFromDeviceArrays(DeviceArray indices, DeviceArray values, long N) throws UnsupportedTypeException {

        if(indices.getArraySize() != values.getArraySize()){
            throw UnsupportedTypeException.create(new Object[]{indices, values, N},"Indices and Values array must have the same size.");
        }

        return new SparseVector(this.grCUDAExecutionContext, indices, values, N);
    }

}
