package com.nvidia.grcuda.runtime.array;

import org.graalvm.polyglot.Context;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.interop.TruffleObject;

public class DenseMatrix implements TruffleObject {
    private final long numElements;

    private final long rows;
    private final long cols;
    private final CUSPARSERegistry.CUSPARSEOrder order;

    private final DeviceArray values;
    private final CUSPARSERegistry.CUDADataType dataType;
    private final UnsafeHelper.Integer64Object dnMatDescr;
    private final boolean isComplex;

    public DenseMatrix(AbstractGrCUDAExecutionContext grCUDAExecutionContext, DeviceArray values, long rows, long cols, CUSPARSERegistry.CUSPARSEOrder order, boolean isComplex) {
        this.values = values;
        this.rows = rows;
        this.cols = cols;
        this.order = order;
        this.isComplex = isComplex;
        numElements = isComplex ? values.getArraySize() / 2 : values.getArraySize();
        dataType = CUSPARSERegistry.CUDADataType.fromGrCUDAType(values.getElementType(), isComplex);

        dnMatDescr = UnsafeHelper.createInteger64Object();

        Context polyglot = Context.getCurrent();
        polyglot.eval("grcuda", "SPARSE::cusparseCreateDnMat").execute(
                dnMatDescr.getAddress(),
                rows,
                cols,
                order == CUSPARSERegistry.CUSPARSEOrder.CUSPARSE_ORDER_ROW ? cols : rows,
                values,
                dataType.ordinal(),
                order);
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

    public UnsafeHelper.Integer64Object getDnMatDescr() {
        return dnMatDescr;
    }

    public boolean isComplex() {
        return isComplex;
    }

    public long getRows() {
        return rows;
    }

    public long getCols() {
        return cols;
    }

    public CUSPARSERegistry.CUSPARSEOrder getOrder() {
        return order;
    }

    public void freeMemory() {
        values.freeMemory();
        dnMatDescr.close();
    }
}
