package com.nvidia.grcuda.functions;

import com.nvidia.grcuda.runtime.array.DenseVector;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.SparseMatrixCSR;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.ExportLibrary;

@ExportLibrary(InteropLibrary.class)
public class DenseVectorFunction extends Function {
    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    private final int NUM_ARGUMENTS = 2;

    public DenseVectorFunction(AbstractGrCUDAExecutionContext grCUDAExecutionContext) {
        super("DenseVector");
        this.grCUDAExecutionContext = grCUDAExecutionContext;
    }

// private boolean isDeviceArrayConstructor(Object[] arguments){
// return arguments[0] instanceof DeviceArray && arguments[1] instanceof DeviceArray;
// }

    @Override
    public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
        if (arguments.length != NUM_ARGUMENTS) {
            throw ArityException.create(NUM_ARGUMENTS, arguments.length);
        }

        DeviceArray values = (DeviceArray) arguments[0];
        boolean isComplex = (Boolean) arguments[1];

        return new DenseVector(grCUDAExecutionContext, values, isComplex);
    }

}
