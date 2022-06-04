package com.nvidia.grcuda.functions;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.SparseMatrixCSR;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.ExportLibrary;

@ExportLibrary(InteropLibrary.class)
public class SparseMatrixCSRFunction extends Function {
    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    private final int NUM_ARGUMENTS = 5;

    public SparseMatrixCSRFunction(AbstractGrCUDAExecutionContext grCUDAExecutionContext) {
        super("SparseMatrixCSR");
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
        DeviceArray cumulativeNnz = (DeviceArray) arguments[1];
        DeviceArray nnzValues = (DeviceArray) arguments[2];
        long dimRow = expectLong(arguments[3]);
        long dimCol = expectLong(arguments[4]);

        return new SparseMatrixCSR(grCUDAExecutionContext, colIndices, cumulativeNnz, nnzValues, dimRow, dimCol);
    }

}
