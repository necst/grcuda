/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
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
package com.nvidia.grcuda.cudalibraries.cusparse;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectLong;

import java.util.ArrayList;
import java.util.Arrays;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.Namespace;
import com.nvidia.grcuda.cudalibraries.CUDALibraryFunction;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.computation.CUDALibraryExecution;
import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.CompilationFinal;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CUSPARSERegistry {
    // TODO: set library directory
    public static final String DEFAULT_LIBRARY = (System.getenv("LIBCUSPARSE_DIR") != null ? System.getenv("LIBCUSPARSE_DIR") : "") + "libcusparse.so";
    // TODO: edit install.sh -> source file (OptionsDescriptor)
    public static final String DEFAULT_LIBRARY_HINT = " (CuSPARSE library location can be set via the --grcuda.CuSPARSELibrary= option. " +
                    "CuSPARSE support can be disabled via --grcuda.CuSPARSEEnabled=false.";
    public static final String NAMESPACE = "SPARSE";

    private final GrCUDAContext context;
    private final String libraryPath;

    @CompilationFinal private TruffleObject cusparseCreateFunction;
    @CompilationFinal private TruffleObject cusparseDestroyFunction;
    @CompilationFinal private TruffleObject cusparseCreateCooFunction;
    @CompilationFinal private TruffleObject cusparseCreateCsrFunction;
    @CompilationFinal private TruffleObject cusparseSpMV_buffersizeFunction;
    @CompilationFinal private TruffleObject cusparseSpMVFunction;
    @CompilationFinal private TruffleObject cusparseCreateDnVecFunction;
    @CompilationFinal private TruffleObject cusparseCreateFunctionNFI;
    @CompilationFinal private TruffleObject cusparseDestroyFunctionNFI;
    @CompilationFinal private TruffleObject cusparseCreateCooFunctionNFI;
    @CompilationFinal private TruffleObject cusparseCreateCsrFunctionNFI;
    @CompilationFinal private TruffleObject cusparseSpMV_buffersizeFunctionNFI;
    @CompilationFinal private TruffleObject cusparseSpMVFunctionNFI;
    @CompilationFinal private TruffleObject cusparseCreateDnVecFunctionNFI;

    private Long cusparseHandle = null;

    public CUSPARSERegistry(GrCUDAContext context) {
        this.context = context;
        // created field in GrCUDAOptions
        libraryPath = context.getOption(GrCUDAOptions.CuSPARSELibrary);
    }

    public void ensureInitialized() {
        if (cusparseHandle == null) {
            CompilerDirectives.transferToInterpreterAndInvalidate();

            // create NFI function objects for handle creation and destruction

            cusparseCreateFunctionNFI = CUSPARSE_CUSPARSECREATE.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseDestroyFunctionNFI = CUSPARSE_CUSPARSEDESTROY.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseCreateCooFunctionNFI = CUSPARSE_CUSPARSECREATECOO.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseCreateCsrFunctionNFI = CUSPARSE_CUSPARSECREATECSR.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseCreateDnVecFunctionNFI = CUSPARSE_CUSPARSECREATEDNVEC.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseSpMV_buffersizeFunctionNFI = CUSPARSE_CUSPARSESPMV_BUFFERSIZE.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseSpMVFunctionNFI = CUSPARSE_CUSPARSESPMV.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);

            public enum cusparseIndexType_t{
                CUSPARSE_INDEX_16U,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_64I
            };

            public enum cusparseIndexBase_t{
                CUSPARSE_INDEX_BASE_ONE,
                CUSPARSE_INDEX_BASE_ZERO
            };

            public enum cudaDataType{
                CUDA_C_16F, // 16 bit complex
                CUDA_C_32F, // 32 bit complex
                CUDA_C_64F, // 64 bit complex
                CUDA_C_8I,  // 8 bit complex as a pair of signed integers
                CUDA_C_8U,  // 8 bit complex as a pair of signed integers
                CUDA_R_16F, // 16 bit real
                CUDA_R_32F, // 32 bit real
                CUDA_R_64F, // 64 bit real
                CUDA_R_8I, 	// 8 bit real as a signed integer
                CUDA_R_8U   //8 bit real as a signed integer
            };

            public enum cusparseOperation_t{
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_TRANSPOSE,
                CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
            };

            public enum cusparseSpMVAlg_t{
                CUSPARSE_SPMV_ALG_DEFAULT,
                CUSPARSE_SPMV_COO_ALG1,
                CUSPARSE_SPMV_COO_ALG2,
                CUSPARSE_SPMV_CSR_ALG1,
                CUSPARSE_SPMV_CSR_ALG2
            };

            // cusparseStatus_t cusparseCreate(cusparseHandle_t handle)

            cusparseCreateFunction = new Function(CUSPARSE_CUSPARSECREATE.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException {
                    checkArgumentLength(arguments, 0);
                    try (UnsafeHelper.Integer64Object handle = UnsafeHelper.createInteger64Object()) {
                        Object result = INTEROP.execute(cusparseCreateFunctionNFI, handle.getAddress());
                        checkCUSPARSEReturnCode(result, "cusparseCreate");
                        return handle.getValue();
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            // cusparseStatus_t cusparseDestroy(cusparseHandle_t* handle)

            cusparseDestroyFunction = new Function(CUSPARSE_CUSPARSEDESTROY.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 1);
                    long handle = expectLong(arguments[0]);
                    try {
                        Object result = INTEROP.execute(cusparseDestroyFunctionNFI, handle);
                        checkCUSPARSEReturnCode(result, "cusparseDestroy");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            // cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr,
            //                  int64_t               rows,
            //                  int64_t               cols,
            //                  int64_t               nnz,
            //                  void*                 cooRowInd,
            //                  void*                 cooColInd,
            //                  void*                 cooValues,
            //                  cusparseIndexType_t   cooIdxType,
            //                  cusparseIndexBase_t   idxBase,
            //                  cudaDataType          valueType)

            cusparseCreateCooFunction = new Function(CUSPARSE_CUSPARSECREATECOO.getName()) {
                @Override
                @TruffleBoundary
                UnsafeHelper.Integer64Object cusparseSpMatDescr = UnsafeHelper.createInteger64Object();
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 10);
                    try{
                        cusparseSpMatDescr = expectLong(arguments[0]); // è un puntatore
                        long rows = expectLong(arguments[1]);
                        long cols = expectLong(arguments[2]);
                        long nnz = expectLong(arguments[3]);
                        long cooRowInd = expectLong(arguments[4]); // puntatori void???
                        long cooColInd = expectLong(arguments[5]);
                        long cooValues = expectLong(arguments[6]);
                        cusparseIndexType_t cooIdxType = cusparseIndexType_t.expectInt(arguments[7]);
                        cusparseIndexBase_t idxBase = cusparseIndexBase_t.expectInt(arguments[8]);
                        cudaDataType valueType = cudaDataType.expectInt(arguments[9]);
                        Object result = INTEROP.execute(cusparseCreateCooFunctionNFI, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, idxBase, valueType);
                        checkCUSPARSEReturnCode(result, "cusparseCreateCoo");
                        return result;
                    } catch (InteropException e){
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            // cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr,
            //                  int64_t               rows,
            //                  int64_t               cols,
            //                  int64_t               nnz,
            //                  void*                 csrRowOffsets,
            //                  void*                 csrColInd,
            //                  void*                 csrValues,
            //                  cusparseIndexType_t   csrRowOffsetsType,
            //                  cusparseIndexType_t   csrColIndType,
            //                  cusparseIndexBase_t   idxBase,
            //                  cudaDataType          valueType)

            cusparseCreateCsrFunction = new Function(CUSPARSE_CUSPARSECREATECSR.getName()) {
                @Override
                @TruffleBoundary
                UnsafeHelper.Integer64Object cusparseSpMatDescr = UnsafeHelper.createInteger64Object();
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 10);
                    try{
                        cusparseSpMatDescr = expectLong(arguments[0]); // è giusto con expectLong?
                        long rows = expectLong(arguments[1]);
                        long cols = expectLong(arguments[2]);
                        long nnz = expectLong(arguments[3]);
                        long csrRowInd = expectLong(arguments[4]);
                        long csrColInd = expectLong(arguments[5]);
                        long csrValues = expectLong(arguments[6]);
                        cusparseIndexType_t cooIdxType = cusparseIndexType_t.expectInt(arguments[7]);
                        cusparseIndexBase_t idxBase = cusparseIndexBase_t.expectInt(arguments[8]);
                        cudaDataType valueType = cudaDataType.expectInt(arguments[9]);
                        Object result = INTEROP.execute(cusparseCreateCsrFunctionNFI, rows, cols, nnz, csrRowInd, csrColInd, csrValues, cooIdxType, idxBase, valueType);
                        checkCUSPARSEReturnCode(result, "cusparseCreateCsr");
                        return result;
                    } catch (InteropException e){
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            //cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
            //                    int64_t               size,
            //                    void*                 values,
            //                    cudaDataType          valueType)

            cusparseCreateDnVecFunction = new Function(CUSPARSE_CUSPARSECREATEDNVEC.getName()) {
                @Override
                @TruffleBoundary
                UnsafeHelper.Integer64Object cusparseDnVecDescr = UnsafeHelper.createInteger64Object();
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 4);
                    try{
                        cusparseDnVecDescr = expectLong(arguments[0]); // è giusto con expectLong?
                        long size = expectLong(arguments[1]);
                        long values = expectLong(arguments[2]);
                        cudaDataType valueType = expectInt(arguments[3];)
                        Object result = INTEROP.execute(cusparseCreateDnVecFunctionNFI, size, values, valueType);
                        checkCUSPARSEReturnCode(result, "cusparseCreateDnVec");
                        return result;
                    } catch (InteropException e){
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            // cusparseSpMV_buffersize: cusparseSpMV_buffersize(cusparseHandle_t handle, cusparseOperation_t opA,
            //                                                                      const void* alpha, cusparseSpMatDescr_t matA,
            //                                                                      cusparseDnVecDescr_t vecX, const void* beta,
            //                                                                      cusparseDnVecDescr_t vecY, cudaDataType computeType,
            //                                                                      cusparseSpMVAlg_t alg, size_t* bufferSize)

            // enum structures necessary for SpMV_buffersize:

            cusparseSpMV_buffersizeFunction = new Function(CUSPARSE_CUSPARSESPMV_BUFFERSIZE.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException {
                    checkArgumentLength(arguments, 10);
                    try {
                        UnsafeHelper.Integer64Object cusparseSpMatDescr = UnsafeHelper.createInteger64Object(); // ho creato una roba da 8 byte
                        UnsafeHelper.Integer64Object vecX = UnsafeHelper.createInteger64Object(); // ho creato una roba da 8 byte
                        UnsafeHelper.Integer64Object vecY = UnsafeHelper.createInteger64Object(); // ho creato una roba da 8 byte
                        long handle = expectLong(arguments[0]);
                        cusparseOperation_t opA = cusparseOperation_t.(arguments[1]);
                        long alpha = expectLong(arguments[2]); // è solo uno scalare, dovrebbe andare così, no?
                        cusparseSpMatDescr = expectLong(arguments[3]);
                        vecX = expectLong(arguments[4]); // non sono sicura
                        long beta = expectLong(arguments[5]);
                        vecY = expectLong(arguments[4]); // neanche qua
                        cudaDataType computeType = cudaDataType.expectInt(arguments(7));
                        cusparseSpMVAlg_t alg = cusparseSpMVAlg_t.expectInt(arguments[8]);
                        long bufferSize = expectLong(arguments[9]);
                        Object result = INTEROP.execute(cusparseSpMV_buffersizeFunctionNFI, handle, opA, alpha, cusparseSpMatDescr, vecX, beta, vecY, computeType, alg, bufferSize);
                        checkCUSPARSEReturnCode(result, "cusparseSpMV_buffersize");
                        return result;
                    } catch (InteropException e){
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            // cusparseStatus_t cusparseSpMV(cusparseHandle_t     handle,
            //             cusparseOperation_t  opA,
            //             const void*          alpha,
            //             cusparseSpMatDescr_t matA,
            //             cusparseDnVecDescr_t vecX,
            //             const void*          beta,
            //             cusparseDnVecDescr_t vecY,
            //             cudaDataType         computeType,
            //             cusparseSpMVAlg_t    alg,
            //             void*                externalBuffer)

            cusparseSpMVFunction = new Function(CUSPARSE_CUSPARSESPMV.getName()){
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException{
                    checkArgumentLength(arguments, 10);
                    try{
                        UnsafeHelper.Integer64Object matA = UnsafeHelper.createInteger64Object(); // ho creato una roba da 8 byte
                        UnsafeHelper.Integer64Object vecX = UnsafeHelper.createInteger64Object(); // ho creato una roba da 8 byte
                        UnsafeHelper.Integer64Object vecY = UnsafeHelper.createInteger64Object(); // ho creato una roba da 8 byte
                        long handle = expectLong(arguments[0]);
                        cusparseOperation_t opA = cusparseOperation_t.(arguments[1]);
                        long alpha = expectLong(arguments[2]); // puntatore a scalare
                        matA = expectLong(arguments[3]);
                        vecX = expectLong(arguments[4]); // non sono sicura
                        long beta = expectLong(arguments[5]);
                        vecY = expectLong(arguments[4]); // neanche qua
                        cudaDataType computeType = cudaDataType.expectInt(arguments(7));
                        cusparseSpMVAlg_t alg = cusparseSpMVAlg_t.expectInt(arguments[8]);
                        long bufferSize = expectLong(arguments[9]);
                        Object result = INTEROP.execute(cusparseSpMVFunctionNFI, handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize);
                        checkCUSPARSEReturnCode(result, "cusparseSmPV");
                        return result;
                    }

                }
            }


            try {
                Object result = INTEROP.execute(cusparseCreateFunction);
                cusparseHandle = expectLong(result);

                context.addDisposable(this::cuSPARSEShutdown);
            } catch (InteropException e) {
                throw new GrCUDAInternalException(e);
            }
        }

    }

    private void cuSPARSEShutdown() {
        CompilerAsserts.neverPartOfCompilation();
        if (cusparseHandle != null) {
            try {
                Object result = InteropLibrary.getFactory().getUncached().execute(cusparseDestroyFunction, cusparseHandle);
                checkCUSPARSEReturnCode(result, CUSPARSE_CUSPARSEDESTROY.getName());
                cusparseHandle = null;
            } catch (InteropException e) {
                throw new GrCUDAInternalException(e);
            }
        }
    }

    public void registerCUSPARSEFunctions(Namespace namespace) {
        // Create function wrappers (decorators for all functions except handle con- and
        // destruction);
        for (ExternalFunctionFactory factory : functions) {
            final Function wrapperFunction = new CUDALibraryFunction(factory.getName(), factory.getNFISignature()) {

                private Function nfiFunction;

                @Override
                @TruffleBoundary
                protected Object call(Object[] arguments) {
                    ensureInitialized();

                    try {
                        if (nfiFunction == null) {
                            CompilerDirectives.transferToInterpreterAndInvalidate();
                            nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
                        }
                        Object result = new CUDALibraryExecution(context.getGrCUDAExecutionContext(), nfiFunction,
                                        this.createComputationArgumentWithValueList(arguments, cusparseHandle)).schedule();
                        checkCUSPARSEReturnCode(result, nfiFunction.getName());
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };
            namespace.addFunction(wrapperFunction);
        }
    }

    private static void checkCUSPARSEReturnCode(Object result, String... function) {
        CompilerAsserts.neverPartOfCompilation();
        int returnCode;
        try {
            returnCode = InteropLibrary.getFactory().getUncached().asInt(result);
        } catch (UnsupportedMessageException e) {
            throw new GrCUDAInternalException("expected return code as Integer object in " + Arrays.toString(function) + ", got " + result.getClass().getName());
        }
        if (returnCode != 0) {
            throw new GrCUDAException(returnCode, cusparseReturnCodeToString(returnCode), function);
        }
    }

    private static String cusparseReturnCodeToString(int returnCode) {
        switch (returnCode) {
            case 0:
                return "CUSPARSE_STATUS_SUCCESS";
            case 1:
                return "CUSPARSE_STATUS_NOT_INITIALIZED";
            case 3:
                return "CUSPARSE_STATUS_ALLOC_FAILED";
            case 7:
                return "CUSPARSE_STATUS_INVALID_VALUE";
            case 8:
                return "CUSPARSE_STATUS_ARCH_MISMATCH";
            case 11:
                return "CUSPARSE_STATUS_MAPPING_ERROR";
            case 13:
                return "CUSPARSE_STATUS_EXECUTION_FAILED";
            case 14:
                return "CUSPARSE_STATUS_INTERNAL_ERROR";
            case 15:
                return "CUSPARSE_STATUS_NOT_SUPPORTED";
            case 16:
                return "CUSPARSE_STATUS_LICENSE_ERROR";
            default:
                return "unknown error code: " + returnCode;
        }
    }

    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATE = new ExternalFunctionFactory("cusparseCreate", "cusparseCreate_v2", "(pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSEDESTROY = new ExternalFunctionFactory("cusparseDestroy", "cusparseDestroy_v2", "(sint64): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATECOO = new ExternalFunctionFactory("cusparseCreateCoo", "cusparseCreateCoo_v2", "(pointer, sint64, " +
                                                                                                        "sint64, sint64, pointer, pointer, pointer, sint32, sint32, sint32): sint32");
                                                                                                        // gli input delle enum sono sint32, giusto?
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATECSR = new ExternalFunctionFactory("cusparseCreateCsr", "cusparseCreateCoo_v2", "(pointer, sint64, sint64, sint64," +
                                                                                                            "pointer, pointer, pointer, sint32, sint32, sint32, sint32): sint32");
                                                                                                        // gli input delle enum sono sint32, giusto?
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATEDNVEC = new ExternalFunctionFactory("cusparseCreateDnVec", "cusparseCreateDnVec_v2", "(pointer, sint64, pointer, " +
                                                                                                                "sint32): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESPMV_BUFFERSIZE = new ExternalFunctionFactory("cusparseSpMV_buffersize", "cusparseSpMV_buffersize_v2", "(sint64," +
                                                                                                                "sint32, pointer, sint64, sint64, pointer, sint64, sint32, sint32, pointer): sint32");
                                                                                                        // sicuramente c'è un problema con i sint64, perchè non vuole i pointer di MatDesc, ma gli oggetti
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESPMV = new ExternalFunctionFactory("cusparseSpMV", "cusparseSpMV_v2", "(sint64, sint32, pointer, sint64, " +
                                                                                                                "sint64, pointer, sint64, sint32, sint32, pointer): sint32");
                                                                                                        // stesso problema con i Desc

    private static final ArrayList<ExternalFunctionFactory> functions = new ArrayList<>();

    }
}
