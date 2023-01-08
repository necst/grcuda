/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.nvidia.grcuda.runtime.array;

import com.nvidia.grcuda.MemberSet;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

@ExportLibrary(InteropLibrary.class)
public class SparseMatrixCSR extends SparseMatrix {

    private static final String UNSUPPORTED_WRITE_ON_SPARSE_DS = "unsupported write on sparse data structures";
    private static final String UNSUPPORTED_DIRECT_READ_ON_SPARSE_DS = "unsupported direct read on sparse data structures";


    /**
     * Attributes to be exported
     */
    protected static final String FREE = "free";
    protected static final String IS_MEMORY_FREED = "isMemoryFreed";
    protected static final String VALUES = "values";
    protected static final String ROW_CUMULATIVE = "cumulativeNnz";
    protected static final String COL_INDICES = "colIndices";
    protected static final String SPMV = "SpMV";
    protected static final String SPGEMM = "SpGEMM";
    protected static final String TRACE = "trace";

    final ValueProfile profile = ValueProfile.createClassProfile();

    /**
     * Column and row-cumulative indices of nnz elements.
     */
    private DeviceArray csrRowOffsets;
    private DeviceArray csrColInd;

    private final CUSPARSERegistry.CUSPARSEIndexType rType;
    private final CUSPARSERegistry.CUSPARSEIndexType cType;


    protected static final MemberSet MEMBERS = new MemberSet(FREE, SPMV, SPGEMM, IS_MEMORY_FREED, VALUES, ROW_CUMULATIVE, COL_INDICES);

    public SparseMatrixCSR(AbstractGrCUDAExecutionContext grCUDAExecutionContext, Object csrColInd, Object csrRowOffsets, Object csrValues,
                           CUSPARSERegistry.CUDADataType dataType,
                           CUSPARSERegistry.CUSPARSEIndexType rType, CUSPARSERegistry.CUSPARSEIndexType cType,
                           long rows, long cols, boolean isComplex) {
        super(grCUDAExecutionContext, csrValues, rows, cols, dataType, isComplex);
        if (csrValues == null) {
            this.csrRowOffsets = new DeviceArray(grCUDAExecutionContext, rows+1, Type.SINT32);
            this.csrColInd = null;
        } else {
            this.csrRowOffsets = (DeviceArray) csrRowOffsets;
            this.csrColInd = (DeviceArray) csrColInd;
        }

        Context polyglot = Context.getCurrent();
        Value cusparseCreateCsrFunction = polyglot.eval("grcuda", "SPARSE::cusparseCreateCsr");

        this.rType = rType;
        this.cType = cType;

        Value resultCsr = cusparseCreateCsrFunction.execute(
                spMatDescr.getAddress(),
                rows,
                cols,
                numElements,
                csrRowOffsets,
                csrColInd,
                csrValues,
                rType.ordinal(),
                cType.ordinal(),
                CUSPARSERegistry.CUSPARSEIndexBase.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
                dataType.ordinal());
    }

    final public long getSizeBytes() {
        checkFreeMatrix();
        return getValues().getSizeBytes() + csrRowOffsets.getSizeBytes() + csrColInd.getSizeBytes();
    }

    public void setCsrRowOffsets(DeviceArray csrRowOffsets) {
        this.csrRowOffsets = csrRowOffsets;
    }

    public void setCsrColInd(DeviceArray csrColInd) {
        this.csrColInd = csrColInd;
    }

    private int binarySearch(DeviceArray array, int start, int end, int target) {
        try {
            while (start <= end) {
                int mid = (start + end) / 2;
                if ((Integer)array.readArrayElement(mid, profile) == target){
                    return mid;
                }
                if ((Integer)array.readArrayElement(mid, profile) < target) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    
        return -1;
    }
    
    private Float trace() {
        float sum = 0;
    
        try {
            for (int i = 0; i < csrRowOffsets.getArraySize() - 1; ++i) {
                Object start = csrRowOffsets.readArrayElement(i, profile);
                Object end = csrRowOffsets.readArrayElement(i+1, profile);
                
                int index = binarySearch(csrColInd, (Integer)start, (Integer)end, i);
                if (index >= 0) {
                    sum += (Float)getValues().readArrayElement(index, profile);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    
        return Float.valueOf(sum);
    }

    @ExportMessage
    final long getArraySize() throws UnsupportedMessageException {
        return getValues().getArraySize() + csrRowOffsets.getArraySize() + csrColInd.getArraySize();
    }

    @Override
    public String toString() {
        return "SparseMatrixCSR{" +
                "matrixFreed=" + matrixFreed +
                ", dimRows=" + rows +
                ", dimCols=" + cols +
                ", cumulativeNnz=" + csrRowOffsets +
                ", colIndices=" + csrColInd +
                ", values=" + getValues() +
                '}';
    }

    public DeviceArray getCsrRowOffsets() {
        return csrRowOffsets;
    }

    public DeviceArray getCsrColInd() {
        return csrColInd;
    }

    protected void finalize() throws Throwable {
        if (!matrixFreed) {
            this.freeMemory();
        }
        super.finalize();
    }

    public void freeMemory() {
        checkFreeMatrix();
        super.freeMemory();
        csrRowOffsets.freeMemory();
        csrColInd.freeMemory();
        matrixFreed = true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return MEMBERS;
    }


    @ExportMessage
    final boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    final Object readArrayElement(long index) throws UnsupportedMessageException, InvalidArrayIndexException { return null; }

    @ExportMessage
    final boolean isArrayElementReadable(long index) { return false; }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberReadable(String memberName,
                             @Cached.Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) {
        String name = memberProfile.profile(memberName);
        return FREE.equals(name) || SPMV.equals(name) || SPGEMM.equals(name) || IS_MEMORY_FREED.equals(name) || VALUES.equals(name) || ROW_CUMULATIVE.equals(name) || COL_INDICES.equals(name) || TRACE.equals(name);
    }

    @ExportMessage
    Object readMember(String memberName,
                      @Cached.Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) throws UnknownIdentifierException {
        if (!isMemberReadable(memberName, memberProfile)) {
            CompilerDirectives.transferToInterpreter();
            throw UnknownIdentifierException.create(memberName);
        }
        if (FREE.equals(memberName)) {
            return new SparseMatrixFreeFunction();
        }

        if (SPMV.equals(memberName)) {
            return new SparseMatrixCSRSpMVFunction();
        }

        if (SPGEMM.equals(memberName)) {
            return new SparseMatrixCSRSpGEMMFunction();
        }

        if (IS_MEMORY_FREED.equals(memberName)) {
            return matrixFreed;
        }

        if(VALUES.equals(memberName)){
            return getValues();
        }

        if(COL_INDICES.equals(memberName)){
            return getCsrColInd();
        }

        if(ROW_CUMULATIVE.equals(memberName)){
            return getCsrRowOffsets();
        }

        if (TRACE.equals(memberName)) {
            return trace();
        }

        CompilerDirectives.transferToInterpreter();
        throw UnknownIdentifierException.create(memberName);
    }


    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberInvocable(String memberName) {
        return FREE.equals(memberName) || SPMV.equals(memberName) || SPGEMM.equals(memberName);
    }

    @ExportMessage
    Object invokeMember(String memberName,
                        Object[] arguments,
                        @CachedLibrary("this") InteropLibrary interopRead,
                        @CachedLibrary(limit = "1") InteropLibrary interopExecute)
            throws UnsupportedTypeException, ArityException, UnsupportedMessageException, UnknownIdentifierException {
        return interopExecute.execute(interopRead.readMember(this, memberName), arguments);
    }

    @ExportLibrary(InteropLibrary.class)
    final class SparseMatrixCSRSpMVFunction implements TruffleObject {
        @ExportMessage
        boolean isExecutable() {
            return true;
        }

        @ExportMessage
        Object execute(Object[] arguments) throws ArityException {
            checkFreeMatrix();
            if (arguments.length != 4) {
                CompilerDirectives.transferToInterpreter();
                throw ArityException.create(4, arguments.length);
            }
            
            Context polyglot = Context.getCurrent();
            Object alpha = arguments[0];
            Object beta = arguments[1];

            Object dnVec = arguments[2];
            Object outVec = arguments[3];

            Value cusparseSpMV = polyglot.eval("grcuda", "SPARSE::cusparseSpMV");

            cusparseSpMV.execute(
                CUSPARSERegistry.CUSPARSEOperation.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
                alpha,
                SparseMatrixCSR.this,
                dnVec,
                dataType.ordinal(),
                beta,
                outVec,
                CUSPARSERegistry.CUSPARSESpMVAlg.CUSPARSE_SPMV_ALG_DEFAULT.ordinal(),
                memoryTracker);

            return outVec;
        }
    }

    @ExportLibrary(InteropLibrary.class)
    final class SparseMatrixCSRSpGEMMFunction implements TruffleObject {
        @ExportMessage
        boolean isExecutable() {
            return true;
        }

        @ExportMessage
        Object execute(Object[] arguments) throws ArityException {
            checkFreeMatrix();
            if (arguments.length != 4 && arguments.length != 3) {
                CompilerDirectives.transferToInterpreter();
                throw ArityException.create(4, arguments.length);
            }

            Context polyglot = Context.getCurrent();
            float alphaVal = Float.valueOf((Integer)arguments[0]);
            float betaVal = Float.valueOf((Integer)arguments[1]);

            SparseMatrixCSR matB = (SparseMatrixCSR) arguments[2];
            SparseMatrixCSR matC;
            if (arguments.length == 4)  matC = (SparseMatrixCSR) arguments[3];
            else matC = new SparseMatrixCSR(
                    getGrCUDAExecutionContext(), null, null, null,
                    matB.getDataType(),
                    CUSPARSERegistry.CUSPARSEIndexType.CUSPARSE_INDEX_32I,
                    CUSPARSERegistry.CUSPARSEIndexType.CUSPARSE_INDEX_32I,
                    matB.getRows(), matB.getCols(), isComplex);

            try (
                UnsafeHelper.Integer64Object descr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object bufferSize1 = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object bufferSize2 = UnsafeHelper.createInteger64Object();

                UnsafeHelper.Integer64Object numRows = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object numCols = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object numNnz = UnsafeHelper.createInteger64Object();

                UnsafeHelper.Float32Object alpha = UnsafeHelper.createFloat32Object();
                UnsafeHelper.Float32Object beta = UnsafeHelper.createFloat32Object()
            ) {
                Value create = polyglot.eval("grcuda", "SPARSE::cusparseSpGEMM_createDescr");
                Value work = polyglot.eval("grcuda", "SPARSE::cusparseSpGEMM_workEstimation");
                Value compute = polyglot.eval("grcuda", "SPARSE::cusparseSpGEMM_compute");
                Value copy = polyglot.eval("grcuda", "SPARSE::cusparseSpGEMM_copy");
                Value destroy = polyglot.eval("grcuda", "SPARSE::cusparseSpGEMM_destroyDescr");
                Value setPointers = polyglot.eval("grcuda", "SPARSE::cusparseCsrSetPointers");
                Value getSize = polyglot.eval("grcuda", "SPARSE::cusparseSpMatGetSize");

                alpha.setValue(alphaVal);
                beta.setValue(betaVal);

                create.execute(descr.getAddress());

                work.execute(0, 0, alpha.getAddress(), SparseMatrixCSR.this.getSpMatDescr().getValue(), matB.getSpMatDescr().getValue(), beta.getAddress(),
                        matC.getSpMatDescr().getValue(), dataType.ordinal(), 0, descr.getValue(), bufferSize1.getAddress(), 0);

                DeviceArray buffer1 = new DeviceArray(getValues().getGrCUDAExecutionContext(), bufferSize1.getValue(), Type.UINT8);

                work.execute(0, 0, alpha.getAddress(), SparseMatrixCSR.this.getSpMatDescr().getValue(), matB.getSpMatDescr().getValue(), beta.getAddress(),
                        matC.getSpMatDescr().getValue(), dataType.ordinal(), 0, descr.getValue(), bufferSize1.getAddress(), buffer1);

                compute.execute(0, 0, alpha.getAddress(), SparseMatrixCSR.this.getSpMatDescr().getValue(), matB.getSpMatDescr().getValue(), beta.getAddress(),
                        matC.getSpMatDescr().getValue(), dataType.ordinal(), 0, descr.getValue(), bufferSize2.getAddress(), 0);

                DeviceArray buffer2 = new DeviceArray(getValues().getGrCUDAExecutionContext(), bufferSize2.getValue(), Type.UINT8);

                compute.execute(0, 0, alpha.getAddress(), SparseMatrixCSR.this.getSpMatDescr().getValue(), matB.getSpMatDescr().getValue(), beta.getAddress(),
                        matC.getSpMatDescr().getValue(), dataType.ordinal(), 0, descr.getValue(), bufferSize2.getAddress(), buffer2);

                getSize.execute(matC.getSpMatDescr().getValue(), numRows.getAddress(), numCols.getAddress(), numNnz.getAddress());

                DeviceArray newColumns = new DeviceArray(getValues().grCUDAExecutionContext, numNnz.getValue(), Type.SINT32);
                DeviceArray newValues = new DeviceArray(getValues().grCUDAExecutionContext, numNnz.getValue(), Type.FLOAT);

                setPointers.execute(matC.getSpMatDescr().getValue(), matC.getCsrRowOffsets(), newColumns, newValues);

                if (matC.getValues() != null)
                    matC.getValues().freeMemory();
                if (matC.getCsrColInd() != null)
                    matC.getCsrColInd().freeMemory();
                matC.setValues(newValues);
                matC.setCsrColInd(newColumns);

                copy.execute(0, 0, alpha.getAddress(), SparseMatrixCSR.this.getSpMatDescr().getValue(), matB.getSpMatDescr().getValue(), beta.getAddress(),
                        matC.getSpMatDescr().getValue(), dataType.ordinal(), 0, descr.getValue());

                destroy.execute(descr.getValue());

                buffer1.freeMemory();
                buffer2.freeMemory();

            } catch (Exception e) {
                e.printStackTrace();
            }

            return matC;
        }
    }
}
