package com.nvidia.grcuda.runtime.array;

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

public abstract class SparseMatrix implements TruffleObject {
    /**
     * Row and column dimensions
     */
    protected final long rows;
    protected final long cols;
    protected long numElements;
    protected final boolean isComplex;

    private DeviceArray values;

    protected CUSPARSERegistry.CUDADataType dataType;
    protected final UnsafeHelper.Integer64Object spMatDescr;

    protected boolean matrixFreed;

    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;

    protected SparseMatrix(AbstractGrCUDAExecutionContext grCUDAExecutionContext, Object values, long rows, long cols, CUSPARSERegistry.CUDADataType dataType, boolean isComplex) {
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.rows = rows;
        this.cols = cols;
        this.isComplex = isComplex;
        this.dataType = dataType;
        if (values == null) {
            this.values = null;
            numElements = 0;
        } else {
            this.values = (DeviceArray) values;
            numElements = isComplex ? this.values.getArraySize() / 2 : this.values.getArraySize();
        }

        spMatDescr = UnsafeHelper.createInteger64Object();
    }

    protected AbstractGrCUDAExecutionContext getGrCUDAExecutionContext() {
        return grCUDAExecutionContext;
    }

    public void setValues(DeviceArray values) {
        this.values = values;
        numElements = isComplex ? values.getArraySize() / 2 : values.getArraySize();
    }

    public DeviceArray getValues() {
        return values;
    }

    public long getRows() {
        return rows;
    }

    public long getCols() {
        return cols;
    }

    public boolean getIsComplex() {
        return isComplex;
    }

    public UnsafeHelper.Integer64Object getSpMatDescr() {
        return spMatDescr;
    }

    public CUSPARSERegistry.CUDADataType getDataType() {
        return dataType;
    }

    protected void checkFreeMatrix() {
        if (matrixFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException("Matrix freed already");
        }
    }

    protected void freeMemory() {
        values.freeMemory();
        spMatDescr.close();
    }

    @ExportLibrary(InteropLibrary.class)
    final class SparseMatrixFreeFunction implements TruffleObject {
        @ExportMessage
        @SuppressWarnings("static-method")
        boolean isExecutable() {
            return true;
        }

        @ExportMessage
        Object execute(Object[] arguments) throws ArityException {
            checkFreeMatrix();
            if (arguments.length != 0) {
                CompilerDirectives.transferToInterpreter();
                throw ArityException.create(0, arguments.length);
            }
            freeMemory();
            return NoneValue.get();
        }
    }
}
