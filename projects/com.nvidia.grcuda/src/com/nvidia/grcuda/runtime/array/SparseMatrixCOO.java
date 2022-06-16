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
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
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
public class SparseMatrixCOO extends SparseMatrix {

    private static final String UNSUPPORTED_WRITE_ON_SPARSE_DS = "unsupported write on sparse data structures";
    private static final String UNSUPPORTED_DIRECT_READ_ON_SPARSE_DS = "unsupported direct read on sparse data structures";

    /**
     * Attributes to be exported
     */
    protected static final String FREE = "free";
    protected static final String IS_MEMORY_FREED = "isMemoryFreed";
    protected static final String VALUES = "values";
    protected static final String ROW_INDICES = "rowIndices";
    protected static final String COL_INDICES = "colIndices";
    protected static final String SPMV = "SpMV";

    /**
     * Row and column indices of nnz elements.
     */
    private final DeviceArray cooRowInd;
    private final DeviceArray cooColInd;


    protected static final MemberSet MEMBERS = new MemberSet(FREE, SPMV, IS_MEMORY_FREED, VALUES, ROW_INDICES, COL_INDICES);

    public SparseMatrixCOO(AbstractGrCUDAExecutionContext grCUDAExecutionContext, DeviceArray cooRowInd, DeviceArray cooColInd, DeviceArray cooValues, long rows, long cols, boolean isComplex) {
        super(cooValues, rows, cols, isComplex);
        this.cooRowInd = cooRowInd;
        this.cooColInd = cooColInd;

        Context polyglot = Context.getCurrent();
        Value cusparseCreateCooFunction = polyglot.eval("grcuda", "SPARSE::cusparseCreateCoo");

        cusparseCreateCooFunction.execute(
                spMatDescr.getAddress(),
                rows,
                cols,
                numElements,
                cooRowInd,
                cooColInd,
                cooValues,
                CUSPARSERegistry.CUSPARSEIndexType.fromGrCUDAType(cooRowInd.getElementType()).ordinal(),
                CUSPARSERegistry.CUSPARSEIndexBase.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
                dataType.ordinal()
        );
    }



    final public long getSizeBytes() {
        checkFreeMatrix();
        return getValues().getSizeBytes() + cooRowInd.getSizeBytes() + cooColInd.getSizeBytes();
    }

    @ExportMessage
    final long getArraySize() throws UnsupportedMessageException {
        return getValues().getArraySize() + cooRowInd.getArraySize() + cooColInd.getArraySize();
    }

    protected void finalize() throws Throwable {
        if (!matrixFreed) {
            this.freeMemory();
        }
        super.finalize();
    }

    public DeviceArray getCooRowInd() {
        return cooRowInd;
    }

    public DeviceArray getCooColInd() {
        return cooColInd;
    }

    @ExportMessage
    final Object readArrayElement(long index) throws UnsupportedMessageException, InvalidArrayIndexException { return null; }

    @ExportMessage
    final boolean isArrayElementReadable(long index) { return false; }


    @Override
    public String toString() {
        return "SparseMatrixCOO{" +
                "matrixFreed=" + matrixFreed +
                ", dimRows=" + rows +
                ", dimCols=" + cols +
                ", rowIndices=" + cooRowInd +
                ", colIndices=" + cooColInd +
                ", values=" + getValues() +
                '}';
    }

    public void freeMemory() {
        checkFreeMatrix();
        super.freeMemory();
        cooRowInd.freeMemory();
        cooColInd.freeMemory();
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
    @SuppressWarnings("static-method")
    boolean isMemberReadable(String memberName,
                             @Cached.Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) {
        String name = memberProfile.profile(memberName);
        return FREE.equals(name) || SPMV.equals(name) || IS_MEMORY_FREED.equals(name) || VALUES.equals(name) || ROW_INDICES.equals(name) || COL_INDICES.equals(name);
    }

    public boolean isMatrixFreed() {
        return matrixFreed;
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
            return new SparseMatrixCOOSpMVFunction();
        }

        if (IS_MEMORY_FREED.equals(memberName)) {
            return isMatrixFreed();
        }

        if(VALUES.equals(memberName)){
            return getValues();
        }

        if(COL_INDICES.equals(memberName)){
            return getCooColInd();
        }

        if(ROW_INDICES.equals(memberName)){
            return getCooRowInd();
        }

        CompilerDirectives.transferToInterpreter();
        throw UnknownIdentifierException.create(memberName);
    }


    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberInvocable(String memberName) {
        return FREE.equals(memberName) || SPMV.equals(memberName);
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
    final class SparseMatrixCOOSpMVFunction implements TruffleObject {
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
                    SparseMatrixCOO.this,
                    dnVec,
                    dataType.ordinal(),
                    beta,
                    outVec,
                    CUSPARSERegistry.CUSPARSESpMVAlg.CUSPARSE_SPMV_ALG_DEFAULT.ordinal());

            return outVec;
        }
    }
}
