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
import sun.security.krb5.internal.SeqNumber;

public class CUSPARSERegistry {
    // TODO: set library directory
    public static final String DEFAULT_LIBRARY = (System.getenv("LIBCUSPARSE_DIR") != null ? System.getenv("LIBCUSPARSE_DIR") : "") + "libcusparse.so.11";
    // TODO: edit install.sh -> source file (OptionsDescriptor)
    public static final String DEFAULT_LIBRARY_HINT = " (CuSPARSE library location can be set via the --grcuda.CuSPARSELibrary= option. " +
                    "CuSPARSE support can be disabled via --grcuda.CuSPARSEEnabled=false.";
    public static final String NAMESPACE = "SPARSE";

    private final GrCUDAContext context;
    private final String libraryPath;

    private LibrarySetStreamFunction cusparseLibrarySetStreamFunction;

    @CompilationFinal private TruffleObject cusparseCreateFunction;
    @CompilationFinal private TruffleObject cusparseDestroyFunction;
    @CompilationFinal private TruffleObject cusparseCreateFunctionNFI;
    @CompilationFinal private TruffleObject cusparseDestroyFunctionNFI;

    private Long cusparseHandle = null;

    public enum cusparseIndexType_t{
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
            try {
                Object result = INTEROP.execute(cusparseCreateFunction);
                cusparseHandle = expectLong(result);
                context.addDisposable(this::cuSPARSEShutdown);
            } catch (InteropException e) {
                throw new GrCUDAInternalException(e);
            }
        }

        Object cusparseSetStreamFunctionNFI = null;
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
                        if (nfiFunction == null) {
                            CompilerDirectives.transferToInterpreterAndInvalidate();
                            nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
                        }
                        // Set the other arguments;
                        //TODO: clean up D:
                        List<ComputationArgumentWithValue> computationArgumentsWithValue = new ArrayList<>();
                        if(!factory.getName().contains("SpMV")){
                            List<ComputationArgument> computationArguments = ComputationArgument.parseParameterSignature(factory.getNFISignature());
                            for (int i = 0; i < arguments.length; i++) {
                                computationArgumentsWithValue.add(new ComputationArgumentWithValue(computationArguments.get(i), arguments[i]));
                            }
                        } else {
                            computationArgumentsWithValue = this.createComputationArgumentWithValueList(arguments, cusparseHandle);
                        }
                        Object result = new CUDALibraryExecution(context.getGrCUDAExecutionContext(), nfiFunction, cusparseLibrarySetStreamFunction,
                                computationArgumentsWithValue).schedule();
                        checkCUSPARSEReturnCode(result, nfiFunction.getName());
                        return result;
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

    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATE = new ExternalFunctionFactory("cusparseCreate", "cusparseCreate", "(pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSEDESTROY = new ExternalFunctionFactory("cusparseDestroy", "cusparseDestroy", "(sint64): sint32");
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


    //

    private static final ArrayList<ExternalFunctionFactory> functions = new ArrayList<>();

    static {
        functions.add(CUSPARSE_CUSPARSECREATE);
        functions.add(CUSPARSE_CUSPARSEDESTROY);
        functions.add(CUSPARSE_CUSPARSECREATECOO);
        functions.add(CUSPARSE_CUSPARSECREATECSR);
        functions.add(CUSPARSE_CUSPARSECREATEDNVEC);
        functions.add(CUSPARSE_CUSPARSESPMV_BUFFERSIZE);
        functions.add(CUSPARSE_CUSPARSESPMV);
    }

}