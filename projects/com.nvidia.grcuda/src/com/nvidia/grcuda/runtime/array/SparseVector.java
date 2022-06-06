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

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.MemberSet;
import com.nvidia.grcuda.NoneValue;

import static com.nvidia.grcuda.functions.Function.expectDouble;
import static com.nvidia.grcuda.functions.Function.expectFloat;
import static com.nvidia.grcuda.functions.Function.expectInt;
import static com.nvidia.grcuda.functions.Function.expectLong;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.computation.arraycomputation.DeviceArrayCopyException;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
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
public class SparseVector implements TruffleObject {

    private static final String ACCESSED_FREED_MEMORY_MESSAGE = "memory of array freed";
    private static final String UNSUPPORTED_WRITE_ON_SPARSE_DS = "unsupported write on sparse data structures";
    private static final String UNSUPPORTED_DIRECT_READ_ON_SPARSE_DS = "unsupported direct read on sparse data structures";

    protected static final String FREE = "free";
    protected static final String IS_MEMORY_FREED = "isMemoryFreed";
    protected static final String VALUES = "values";
    protected static final String INDICES = "indices";
    protected static final String GEMVI = "gemvi";


    /**
     * Values of non zero elements stored in the array.
     */
    private boolean vectorFreed;


    /**
     * Inner arrays
     */
    private final DeviceArray indices;
    private final DeviceArray values;

    private final boolean isComplex;
    private final CUSPARSERegistry.CUDADataType dataType;

    /**
     * Number non zero elements stored in the array.
     */
    private final long numNnz;

    /**
     * Array properties
     */
    private final long N;
    private final long sizeBytes;

    /**
     * Handle to cusparse
     */
    private Context polyglot = Context.getCurrent();
    private Value cu = polyglot.eval("grcuda", "CU");

    /**
     * Callable functions or general accessible members
     */
    protected static final MemberSet MEMBERS = new MemberSet(FREE, IS_MEMORY_FREED, VALUES, INDICES, GEMVI);

    public SparseVector(AbstractGrCUDAExecutionContext grCUDAExecutionContext, DeviceArray values, DeviceArray indices, long N, boolean isComplex) {
        this.values = values;
        this.indices = indices;
        this.numNnz = values.getArraySize();
        this.sizeBytes = values.getSizeBytes() + indices.getSizeBytes();
        this.N = N;
        this.isComplex = isComplex;
        this.dataType = CUSPARSERegistry.CUDADataType.fromGrCUDAType(values.getElementType(), isComplex);
    }

    @ExportMessage
    boolean isArrayElementReadable(long index) {
        return !vectorFreed && index >= 0 && index < numNnz;
    }

    @ExportMessage
    boolean isArrayElementModifiable(long index) {
        return index >= 0 && index < numNnz;
    }

    @SuppressWarnings("static-method")
    @ExportMessage
    boolean isArrayElementInsertable(@SuppressWarnings("unused") long index) {
        return false;
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
    @SuppressWarnings("static-method")
    boolean isMemberReadable(String memberName,
                             @Cached.Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) {
        String name = memberProfile.profile(memberName);
        return FREE.equals(name) || IS_MEMORY_FREED.equals(name) || VALUES.equals(name) || INDICES.equals(name) || GEMVI.equals(name);
    }

    @ExportMessage
    Object readMember(String memberName,
                      @Cached.Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) throws UnknownIdentifierException {
        if (!isMemberReadable(memberName, memberProfile)) {
            CompilerDirectives.transferToInterpreter();
            throw UnknownIdentifierException.create(memberName);
        }
        if (FREE.equals(memberName)) {
            return new SparseVectorFreeFunction();
        }

        if(GEMVI.equals(memberName)){
            return new SparseVectorGemviFunction();
        }

        if (IS_MEMORY_FREED.equals(memberName)) {
            return isVectorFreed();
        }

        if(VALUES.equals(memberName)){
            return getValues();
        }

        if(INDICES.equals(memberName)){
            return getIndices();
        }

        CompilerDirectives.transferToInterpreter();
        throw UnknownIdentifierException.create(memberName);
    }


    @ExportMessage
    final boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    final void writeArrayElement(@SuppressWarnings("unused") long index, @SuppressWarnings("unused") Object value) throws GrCUDAException {
        throw new GrCUDAException(UNSUPPORTED_WRITE_ON_SPARSE_DS);
    }


    @ExportMessage
    final Object readArrayElement(@SuppressWarnings("unused") long index) throws GrCUDAException {
        throw new GrCUDAException(UNSUPPORTED_DIRECT_READ_ON_SPARSE_DS);
    }


    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberInvocable(String memberName) {
        return FREE.equals(memberName) || GEMVI.equals(memberName);
    }

    @ExportMessage
    Object invokeMember(String memberName,
                        Object[] arguments,
                        @CachedLibrary("this") InteropLibrary interopRead,
                        @CachedLibrary(limit = "1") InteropLibrary interopExecute)
            throws UnsupportedTypeException, ArityException, UnsupportedMessageException, UnknownIdentifierException {
        return interopExecute.execute(interopRead.readMember(this, memberName), arguments);
    }

    @ExportMessage
    final long getArraySize() throws UnsupportedMessageException {
        return this.N;
    }

    public void freeMemory() {
        checkFreeVector();
        values.freeMemory();
        indices.freeMemory();
        vectorFreed = true;
    }

    private void executeGemvi(int numRows, int numCols, DeviceArray alpha, DeviceArray matA, DeviceArray beta, DeviceArray outVec) {

        char type;
        switch (dataType) {
            case CUDA_R_64F:
                type = 'D';
                break;
            case CUDA_C_32F:
                type = 'C';
                break;
            case CUDA_C_64F:
                type = 'Z';
                break;
            default:
                type = 'S';
        }


        this.polyglot
                .eval("grcuda", "SPARSE::cusparse" + type + "gemvi")
                .execute(
                        CUSPARSERegistry.CUSPARSEOperation.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
                        numRows,
                        numCols,
                        alpha,
                        matA,
                        numRows,
                        this.numNnz,
                        this.values,
                        this.indices,
                        beta,
                        outVec,
                        CUSPARSERegistry.CUSPARSEIndexBase.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
                        type
                );
    }

    private void checkFreeVector() {
        if (vectorFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
    }


    public DeviceArray getIndices() {
        return indices;
    }

    public DeviceArray getValues() {
        return values;
    }

    final public long getSizeBytes() {
        checkFreeVector();
        return sizeBytes;
    }

    public boolean isVectorFreed() {
        return vectorFreed;
    }

    @Override
    public String toString() {
        return "SparseVector{" +
                "vectorFreed=" + vectorFreed +
                ", indices=" + indices +
                ", values=" + values +
                ", numNnz=" + numNnz +
                ", N=" + N +
                ", sizeBytes=" + sizeBytes +
                '}';
    }

    protected void finalize() throws Throwable {
        if (!vectorFreed) {
            this.freeMemory();
        }
        super.finalize();
    }


    @ExportLibrary(InteropLibrary.class)
    final class SparseVectorFreeFunction implements TruffleObject {
        @ExportMessage
        @SuppressWarnings("static-method")
        boolean isExecutable() {
            return true;
        }

        @ExportMessage
        Object execute(Object[] arguments) throws ArityException {
            checkFreeVector();
            if (arguments.length != 0) {
                CompilerDirectives.transferToInterpreter();
                throw ArityException.create(0, arguments.length);
            }
            freeMemory();
            return NoneValue.get();
        }
    }

    @ExportLibrary(InteropLibrary.class)
    final class SparseVectorGemviFunction implements TruffleObject {
        private static final int NUM_ARGS = 6;
        @ExportMessage
        @SuppressWarnings("static-method")
        boolean isExecutable() {
            return true;
        }
        @ExportMessage
        Object execute(Object[] arguments) throws ArityException, UnsupportedTypeException {
            checkFreeVector();
            if (arguments.length != NUM_ARGS) {
                CompilerDirectives.transferToInterpreter();
                throw ArityException.create(NUM_ARGS, arguments.length);
            }
            int row = expectInt(arguments[0]);
            int col = expectInt(arguments[1]);
            DeviceArray alpha = (DeviceArray) arguments[2];
            DeviceArray matA = (DeviceArray) arguments[3];
            DeviceArray beta = (DeviceArray) arguments[4];
            DeviceArray outVec = (DeviceArray) arguments[5];
            executeGemvi(row, col, alpha, matA, beta, outVec);
            return NoneValue.get();
        }
    }
}
