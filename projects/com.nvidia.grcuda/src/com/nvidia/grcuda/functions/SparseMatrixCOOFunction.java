package com.nvidia.grcuda.functions;

import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.SparseMatrixCOO;
import com.nvidia.grcuda.runtime.array.SparseVector;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.ExportLibrary;

@ExportLibrary(InteropLibrary.class)
public class SparseMatrixCOOFunction extends Function {
    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    private final int NUM_ARGUMENTS = 6;

    public SparseMatrixCOOFunction(AbstractGrCUDAExecutionContext grCUDAExecutionContext) {
        super("SparseMatrixCOO");
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

        DeviceArray colIndices = (DeviceArray) arguments[0];
        DeviceArray rowIndices = (DeviceArray) arguments[1];
        DeviceArray nnzValues = (DeviceArray) arguments[2];
        long dimRow = expectLong(arguments[3]);
        long dimCol = expectLong(arguments[4]);
        boolean isComplex = (Boolean) arguments[5];

        return new SparseMatrixCOO(grCUDAExecutionContext, colIndices, rowIndices, nnzValues, dimRow, dimCol, isComplex);
    }

}
