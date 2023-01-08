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
    private final int NUM_ARGUMENTS = 6;
    private final int NUM_ARGUMENTS_NULL = 4;

    public SparseMatrixCSRFunction(AbstractGrCUDAExecutionContext grCUDAExecutionContext) {
        super("SparseMatrixCSR");
        this.grCUDAExecutionContext = grCUDAExecutionContext;
    }

// private boolean isDeviceArrayConstructor(Object[] arguments){
// return arguments[0] instanceof DeviceArray && arguments[1] instanceof DeviceArray;
// }

    @Override
    public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
        if (arguments.length != NUM_ARGUMENTS && arguments.length != NUM_ARGUMENTS_NULL) {
            throw ArityException.create(NUM_ARGUMENTS, NUM_ARGUMENTS, arguments.length);
        }

        if (arguments.length == NUM_ARGUMENTS) {
            DeviceArray colIndices = (DeviceArray) arguments[0];
            DeviceArray cumulativeNnz = (DeviceArray) arguments[1];
            DeviceArray nnzValues = (DeviceArray) arguments[2];
            long dimRow = expectLong(arguments[3]);
            long dimCol = expectLong(arguments[4]);
            boolean isComplex = (Boolean) arguments[5];

            return new SparseMatrixCSR(grCUDAExecutionContext, colIndices, cumulativeNnz, nnzValues,
                    CUSPARSERegistry.CUDADataType.fromGrCUDAType(nnzValues.getElementType(), isComplex),
                    CUSPARSERegistry.CUSPARSEIndexType.fromGrCUDAType(cumulativeNnz.getElementType()),
                    CUSPARSERegistry.CUSPARSEIndexType.fromGrCUDAType(colIndices.getElementType()),
                    dimRow, dimCol, isComplex);
        } else {
            long dimRow = expectLong(arguments[0]);
            long dimCol = expectLong(arguments[1]);
            boolean isComplex = (Boolean) arguments[2];
            String dataType = (String) arguments[3];


            return new SparseMatrixCSR(
                    grCUDAExecutionContext, null, null, null,
                    CUSPARSERegistry.CUDADataType.valueOf(dataType),
                    CUSPARSERegistry.CUSPARSEIndexType.CUSPARSE_INDEX_32I,
                    CUSPARSERegistry.CUSPARSEIndexType.CUSPARSE_INDEX_32I,
                    dimRow, dimCol, isComplex);
        }
    }

}
