package com.nvidia.grcuda.runtime.array;

import org.graalvm.polyglot.Context;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.interop.TruffleObject;

public class DenseVector implements TruffleObject {
    private final long numElements;

    private final DeviceArray values;
    private final CUSPARSERegistry.CUDADataType dataType;
    private final UnsafeHelper.Integer64Object dnVecDescr;
    private final boolean isComplex;

    public DenseVector(AbstractGrCUDAExecutionContext grCUDAExecutionContext, DeviceArray values, boolean isComplex) {
        this.values = values;
        this.isComplex = isComplex;
        numElements = isComplex ? values.getArraySize() / 2 : values.getArraySize();
        dataType = CUSPARSERegistry.CUDADataType.fromGrCUDAType(values.getElementType(), isComplex);

        dnVecDescr = UnsafeHelper.createInteger64Object();

        Context polyglot = Context.getCurrent();
        polyglot.eval("grcuda", "SPARSE::cusparseCreateDnVec").execute(
                dnVecDescr.getAddress(),
                numElements,
                values,
                dataType.ordinal());
    }

    public long getNumElements() {
        return numElements;
    }

    public DeviceArray getValues() {
        return values;
    }

    public CUSPARSERegistry.CUDADataType getDataType() {
        return dataType;
    }

    public UnsafeHelper.Integer64Object getDnVecDescr() {
        return dnVecDescr;
    }

    public boolean isComplex() {
        return isComplex;
    }

    public void freeMemory() {
        values.freeMemory();
        dnVecDescr.close();
    }
}
