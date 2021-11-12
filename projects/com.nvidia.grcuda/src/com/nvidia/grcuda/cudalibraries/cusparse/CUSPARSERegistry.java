/*
 * Copyright (c) 2021, NECSTLab, Politecnico di Milano. All rights reserved.
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
import java.util.List;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.Namespace;
import com.nvidia.grcuda.TypeException;
import com.nvidia.grcuda.cudalibraries.CUDALibraryFunction;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.computation.CUDALibraryExecution;
import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.runtime.stream.CUSPARSESetStreamFunction;
import com.nvidia.grcuda.runtime.stream.LibrarySetStreamFunction;
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
    public static final String DEFAULT_LIBRARY = (System.getenv("LIBCUSPARSE_DIR") != null ? System.getenv("LIBCUSPARSE_DIR") : "") + "libcusparse.so";
    public static final String DEFAULT_LIBRARY_HINT = " (CuSPARSE library location can be set via the --grcuda.CuSPARSELibrary= option. " +
                    "CuSPARSE support can be disabled via --grcuda.CuSPARSEEnabled=false.";
    public static final String NAMESPACE = "SPARSE";

    private final GrCUDAContext context;
    private final String libraryPath;

    private LibrarySetStreamFunction cusparseLibrarySetStreamFunction;

    @CompilationFinal private TruffleObject cusparseCreateFunction;
    @CompilationFinal private TruffleObject cusparseDestroyFunction;
    @CompilationFinal private TruffleObject cusparseSetStreamFunction;
    @CompilationFinal private TruffleObject cusparseCreateCooFunction;
    @CompilationFinal private TruffleObject cusparseCreateCsrFunction;
    @CompilationFinal private TruffleObject cusparseCreateDnVecFunction;
    @CompilationFinal private TruffleObject cusparseSpMV_bufferSizeFunction;
    @CompilationFinal private TruffleObject cusparseSgemvi_bufferSizeFunction;
    @CompilationFinal private TruffleObject cusparseCreateFunctionNFI;
    @CompilationFinal private TruffleObject cusparseDestroyFunctionNFI;
    @CompilationFinal private TruffleObject cusparseSetStreamFunctionNFI;
    @CompilationFinal private TruffleObject cusparseCreateCooFunctionNFI;
    @CompilationFinal private TruffleObject cusparseCreateCsrFunctionNFI;
    @CompilationFinal private TruffleObject cusparseCreateDnVecFunctionNFI;
    @CompilationFinal private TruffleObject cusparseSpMV_bufferSizeFunctionNFI;
    @CompilationFinal private TruffleObject cusparseSgemvi_bufferSizeFunctionNFI;



    private Long cusparseHandle = null;

    public enum cusparseIndexType_t{
        CUSPARSE_INDEX_UNUSED,
        CUSPARSE_INDEX_16U,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_64I;
    }

    public enum cusparseIndexBase_t {
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_INDEX_BASE_ONE;
    }

    public enum cudaDataType {
        CUDA_R_32F, // 32 bit real
        CUDA_R_64F, // 64 bit real
        CUDA_R_16F, // 16 bit real
        CUDA_R_8I, // 8 bit real as a signed integer
        CUDA_C_32F, // 32 bit complex
        CUDA_C_64F, // 64 bit complex
        CUDA_C_16F, // 16 bit complex
        CUDA_C_8I,  // 8 bit complex as a pair of signed integers
        CUDA_R_8U,   //8 bit real as a signed integer
        CUDA_C_8U;  // 8 bit complex as a pair of signed integers
    }

    public enum cusparseOperation_t {
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    }

    public enum cusparseSpMVAlg_t {
        CUSPARSE_SPMV_ALG_DEFAULT,
        CUSPARSE_SPMV_COO_ALG1,
        CUSPARSE_SPMV_COO_ALG2,
        CUSPARSE_SPMV_CSR_ALG1,
        CUSPARSE_SPMV_CSR_ALG2;
    }

    public CUSPARSERegistry(GrCUDAContext context) {
        this.context = context;
        // created field in GrCUDAOptions
        libraryPath = context.getOption(GrCUDAOptions.CuSPARSELibrary);
    }

    public void ensureInitialized() {
        if (cusparseHandle == null) {
            CompilerDirectives.transferToInterpreterAndInvalidate();

            // create NFI function objects for functions' management

            cusparseCreateFunctionNFI = CUSPARSE_CUSPARSECREATE.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseDestroyFunctionNFI = CUSPARSE_CUSPARSEDESTROY.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseSetStreamFunctionNFI = CUSPARSE_CUSPARSESETSTREAM.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseCreateCooFunctionNFI = CUSPARSE_CUSPARSECREATECOO.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseCreateCsrFunctionNFI = CUSPARSE_CUSPARSECREATECSR.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseCreateDnVecFunctionNFI = CUSPARSE_CUSPARSECREATEDNVEC.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseSpMV_bufferSizeFunctionNFI = CUSPARSE_CUSPARSESPMV_BUFFERSIZE.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseSgemvi_bufferSizeFunctionNFI = CUSPARSE_CUSPARSESGEMVI_BUFFERSIZE.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
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

            // cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId)

            cusparseSetStreamFunction = new Function(CUSPARSE_CUSPARSESETSTREAM.getName()) {
                @Override
                @TruffleBoundary
                public Object call (Object[] arguments) throws ArityException {
                    checkArgumentLength(arguments, 2);
                    long handle = expectLong(arguments[0]);
                    long streamId = expectLong(arguments[1]);
                    try {
                        Object result = INTEROP.execute(cusparseSetStreamFunctionNFI, handle, streamId);
                        checkCUSPARSEReturnCode(result, "cusparseSetStream");
                        return result;
                    } catch (InteropException e){
                        throw new GrCUDAInternalException(e);
                    }
                }
            }

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
                Long cusparseSpMatDescr = null;
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 10);
                    cusparseSpMatDescr = expectLong(arguments[0]);
                    long rows = expectLong(arguments[1]);
                    long cols = expectLong(arguments[2]);
                    long nnz = expectLong(arguments[3]);
                    long cooRowIdx = expectLong(arguments[4]);
                    long cooColIdx = expectLong(arguments[5]);
                    long cooValues = expectLong(arguments[6]);
                    cusparseIndexType_t cooIdxType = cusparseIndexType_t.values()[expectInt(arguments[7])];
                    cusparseIndexBase_t cooIdxBase = cusparseIndexBase_t.values()[expectInt(arguments[8])];
                    cudaDataType valueType = cudaDataType.values()[expectInt(arguments[9])];
                    try {
                        Object result = INTEROP.execute(cusparseCreateCooFunctionNFI, rows, cols, nnz, cooRowIdx, cooColIdx, cooValues,
                                cooIdxType.ordinal(), cooIdxBase.ordinal(), valueType.ordinal());
                        checkCUSPARSEReturnCode(result, "cusparseCreateCoo");
                        return result;
                    } catch(InteropException e){
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
                Long cusparseSpMatDescr = null;
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException {
                    checkArgumentLength(arguments, 11);
                    cusparseSpMatDescr = expectLong(arguments[0]);
                    long rows = expectLong(arguments[1]);
                    long cols = expectLong(arguments[2]);
                    long nnz = expectLong(arguments[3]);
                    long csrRowOffsets = expectLong(arguments[4]);
                    long csrColIdx = expectLong(arguments[5]);
                    long csrValues = expectLong(arguments[6]);
                    cusparseIndexType_t csrRowOffsetsType = cusparseIndexType_t.values()[expectInt(arguments[7])];
                    cusparseIndexType_t csrColIdxType = cusparseIndexType_t.values()[expectInt(arguments[8])];
                    cusparseIndexBase_t csrIdxBase = cusparseIndexBase_t.values()[expectInt(arguments[9])];
                    cudaDataType valueType = cudaDataType.values()[expectInt(arguments[10])];
                    try {
                        Object result = INTEROP.execute(cusparseCreateCooFunctionNFI, rows, cols, nnz, csrRowOffsets, csrColIdx, csrValues,
                                csrRowOffsetsType.ordinal(), csrColIdxType.ordinal(), csrIdxBase.ordinal(), valueType.ordinal());
                        checkCUSPARSEReturnCode(result, "cusparseCreateCsr");
                        return result;
                    } catch(InteropException e){
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            // cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
            //                    int64_t               size,
            //                    void*                 values,
            //                    cudaDataType          valueType)

            cusparseCreateDnVecFunction = new Function(CUSPARSE_CUSPARSECREATEDNVEC.getName()) {
                Long cusparseDnVecDescr = null;
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException {
                    checkArgumentLength(arguments, 4);
                    cusparseDnVecDescr = expectLong(arguments[0]);
                    long size = expectLong(arguments[1]);
                    long values = expectLong(arguments[2]);
                    cudaDataType valueType = cudaDataType.values()[expectInt(arguments[3])];
                    try {
                        Object result = INTEROP.execute(cusparseCreateDnVecFunctionNFI, size, values, valueType.ordinal());
                        checkCUSPARSEReturnCode(result, "cusparseCreateDnVec");
                        return result;
                    } catch (InteropException e){
                        throw new GrCUDAInternalException(e);
                    }
                }

            };

            //            cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t     handle,
//                    cusparseOperation_t  opA,
//                        const void*          alpha,
//                    cusparseSpMatDescr_t matA,
//                    cusparseDnVecDescr_t vecX,
//                        const void*          beta,
//                    cusparseDnVecDescr_t vecY,
//                    cudaDataType         computeType,
//                    cusparseSpMVAlg_t    alg,
//                    size_t*              bufferSize)

            cusparseSpMV_bufferSizeFunction = new Function(CUSPARSE_CUSPARSESPMV_BUFFERSIZE.getName()){
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException{
                    checkArgumentLength(arguments, 10);
                    long handle = expectLong(arguments[0]);
                    cusparseOperation_t opA = cusparseOperation_t.values()[expectInt(arguments[1])];
                    long alpha = expectLong(arguments[2]);
                    long cusparseSpMatDesc = expectLong(arguments[3]);
                    long vecX = expectLong(arguments[4]);
                    long beta = expectLong(arguments[5]);
                    long vecY = expectLong(arguments[6]);
                    cudaDataType computeType = cudaDataType.values()[expectInt(arguments[7])];
                    cusparseSpMVAlg_t alg = cusparseSpMVAlg_t.values()[expectInt(arguments[8])];
                    long bufferSize = expectLong(arguments[9]);
                    try{
                        Object result = INTEROP.execute(cusparseSpMV_bufferSizeFunctionNFI, handle, opA, alpha,
                                cusparseSpMatDesc, vecX, beta, vecY, computeType, alg, bufferSize);
                        checkCUSPARSEReturnCode(result, "cusparseSpMV_bufferSize");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            // cusparseStatus_t cusparseSgemvi_bufferSize(cusparseHandle_t handle,
            //                  cusparseOperation_t transA,
            //                  int m,
            //                  int n,
            //                  int nnz,
            //                  int* pBufferSize)

            cusparseSgemvi_bufferSizeFunction = new Function(CUSPARSE_CUSPARSESGEMVI_BUFFERSIZE.getName()){
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException{
                    checkArgumentLength(6);
                    long handle = expectLong(arguments[0]);
                    cusparseOperation_t transA = cusparseOperation_t.values()[expectInt(arguments[1])];
                    int m = expectInt(arguments[2]);
                    int n = expectInt(arguments[3]);
                    int nnz = expectInt(arguments[4]);
                    long pBufferSize = expectLong(arguments[5]);
                    try {
                        Object result = INTEROP.execute(cusparseSgemvi_bufferSizeFunctionNFI, handle, transA, m, n, nnz, pBufferSize);
                        checkCUSPARSEReturnCode(result, "cusparseSgemvi_bufferSize");
                        return result
                    } catch (InteropException e){
                        throw new GrCUDAInternalException(e);
                    }
                }
            };




            try {
                Object result = INTEROP.execute(cusparseCreateFunction);
                cusparseHandle = expectLong(result);
                context.addDisposable(this::cuSPARSEShutdown);
            } catch (InteropException e) {
                throw new GrCUDAInternalException(e);
            }
        }

        cusparseLibrarySetStreamFunction = new CUSPARSESetStreamFunction((Function) cusparseSetStreamFunctionNFI, cusparseHandle);

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
        // Create function wrappers
        for (ExternalFunctionFactory factory : functions) {
            final Function wrapperFunction = new CUDALibraryFunction(factory.getName(), factory.getNFISignature()) {

                private Function nfiFunction;

                @Override
                @TruffleBoundary
                protected Object call(Object[] arguments) {
                    ensureInitialized();

                    try {
                        if (nfiFunction == null){
                            CompilerDirectives.transferToInterpreterAndInvalidate();
                            nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
                        }
                        List<ComputationArgumentWithValue> computationArgumentsWithValue = new ArrayList<>();
                        // proxy

//                        if (nfiFunction == null) {
//                            CompilerDirectives.transferToInterpreterAndInvalidate();
//                            nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
//                        }
//                        // Set the other arguments;
//                        //TODO: clean up D:
//                        List<ComputationArgumentWithValue> computationArgumentsWithValue = new ArrayList<>();
//                        if(!factory.getName().contains("SpMV")){
//                            List<ComputationArgument> computationArguments = ComputationArgument.parseParameterSignature(factory.getNFISignature());
//                            for (int i = 0; i < arguments.length; i++) {
//                                computationArgumentsWithValue.add(new ComputationArgumentWithValue(computationArguments.get(i), arguments[i]));
//                            }
//                        } else {
//                            computationArgumentsWithValue = this.createComputationArgumentWithValueList(arguments, cusparseHandle);
//                        }
//                        Object result = new CUDALibraryExecution(context.getGrCUDAExecutionContext(), nfiFunction, cusparseLibrarySetStreamFunction,
//                                computationArgumentsWithValue).schedule();
//                        checkCUSPARSEReturnCode(result, nfiFunction.getName());
//                        return result;
                    } catch (InteropException | TypeException e) {
                        throw new GrCUDAInternalException((InteropException) e);
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
            case 2:
                return "CUSPARSE_STATUS_ALLOC_FAILED";
            case 3:
                return "CUSPARSE_STATUS_INVALID_VALUE";
            case 4:
                return "CUSPARSE_STATUS_ARCH_MISMATCH";
            case 5:
                return "CUSPARSE_STATUS_EXECUTION_FAILED";
            case 6:
                return "CUSPARSE_STATUS_INTERNAL_ERROR";
            case 7:
                return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
            case 8:
                return "CUSPARSE_STATUS_NOT_SUPPORTED";
            case 9:
                return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
            default:
                return "unknown error code: " + returnCode;
        }
    }

    // functions exposed to the user

    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATE = new ExternalFunctionFactory("cusparseCreate", "cusparseCreate", "(pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSEDESTROY = new ExternalFunctionFactory("cusparseDestroy", "cusparseDestroy", "(sint64): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESETSTREAM = new ExternalFunctionFactory("cusparseSetStream", "cusparseSetStream", "(sint64, sint64): sint32")
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATECOO = new ExternalFunctionFactory("cusparseCreateCoo", "cusparseCreateCoo", "(pointer, sint64, " +
                                                                                                        "sint64, sint64, pointer, pointer, pointer, sint32, sint32, sint32): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATECSR = new ExternalFunctionFactory("cusparseCreateCsr", "cusparseCreateCsr", "(pointer, sint64, sint64, sint64," +
                                                                                                            "pointer, pointer, pointer, sint32, sint32, sint32, sint32): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATEDNVEC = new ExternalFunctionFactory("cusparseCreateDnVec", "cusparseCreateDnVec", "(pointer, sint64, pointer, " +
                                                                                                                "sint32): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESPMV_BUFFERSIZE = new ExternalFunctionFactory("cusparseSpMV_bufferSize", "cusparseSpMV_bufferSize", "(sint64, sint32," +
                                                                                                                "pointer, sint64, sint64, pointer, sint64, sint32, sint32, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESPMV = new ExternalFunctionFactory("cusparseSpMV", "cusparseSpMV", "(sint64, sint32, pointer, sint64, " +
                                                                                                                "sint64, pointer, sint64, sint32, sint32, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESGEMVI_BUFFERSIZE = new ExternalFunctionFactory("cusparseSpGEMM", "cusparseSpGEMM", "(sint64, sint32, sint32, " +
                                                                                                                "sint32, sint32, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESGEMVI = new ExternalFunctionFactory("cusparseSgemvi", "cusparseSgemvi", "(sint64, sint32, sint32, sint32" +
                                                                                                                "pointer, pointer, sint32, sint32, pointer, pointer, pointer, pointer, sint32, pointer): sint32");




    private static final ArrayList<ExternalFunctionFactory> functions = new ArrayList<>();

    static {
//        functions.add(CUSPARSE_CUSPARSECREATE);
//        functions.add(CUSPARSE_CUSPARSEDESTROY);
//        functions.add(CUSPARSE_CUSPARSECREATECOO);
//        functions.add(CUSPARSE_CUSPARSECREATECSR);
//        functions.add(CUSPARSE_CUSPARSECREATEDNVEC);
//        functions.add(CUSPARSE_CUSPARSESPMV_BUFFERSIZE);
        functions.add(CUSPARSE_CUSPARSESPMV);
        functions.add(CUSPARSE_CUSPARSESGEMVI);
    }

}
