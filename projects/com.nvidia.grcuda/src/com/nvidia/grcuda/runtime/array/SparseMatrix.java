package com.nvidia.grcuda.runtime.array;

import java.io.Closeable;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

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
    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    protected final long rows;
    protected final long cols;
    protected final boolean isComplex;
    protected boolean matrixFreed;
    protected CUSPARSERegistry.CUDADataType dataType;

    private DeviceArray values;
    protected long numElements;

    protected final UnsafeHelper.Integer64Object spMatDescr;

    // contains all unsafe objects that need to be eventually freed
    protected final List<Closeable> memoryTracker;

    protected SparseMatrix(AbstractGrCUDAExecutionContext grCUDAExecutionContext, DeviceArray values, long rows, long cols, CUSPARSERegistry.CUDADataType dataType, boolean isComplex) {
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.rows = rows;
        this.cols = cols;
        this.isComplex = isComplex;
        this.dataType = dataType;
        this.values = values;
        if (values == null) {
            numElements = 0;
        } else {
            numElements = isComplex ? this.values.getArraySize() / 2 : this.values.getArraySize();
        }

        spMatDescr = UnsafeHelper.createInteger64Object();
        matrixFreed = false;
        memoryTracker = new LinkedList<>();
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
        spMatDescr.close();
        for (Closeable o : memoryTracker) {
            try {
                o.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        memoryTracker.clear();
    }

    public void track(Closeable c) {
        memoryTracker.add(c);
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
                throw ArityException.create(0, 0, arguments.length);
            }
            freeMemory();
            return NoneValue.get();
        }
    }
}
