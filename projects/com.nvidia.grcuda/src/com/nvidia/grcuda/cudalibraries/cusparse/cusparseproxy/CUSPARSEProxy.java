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
package com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy;

import static com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry.checkCUSPARSEReturnCode;
import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.checkArgumentLength;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.GrCUDAOptionMap;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.CUDAFunction;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CUSPARSEProxy {

    @CompilerDirectives.CompilationFinal private TruffleObject cusparseSetStreamFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseCreateCooFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseCreateCsrFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseCreateDnVecFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseCreateDnMatFunctionNFI;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSpMV_bufferSizeFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseSpVV_bufferSizeFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseSpMM_bufferSizeFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseSgemvi_bufferSizeFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseCgemvi_bufferSizeFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseDgemvi_bufferSizeFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseZgemvi_bufferSizeFunctionNFI;

    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseCreateCooFunction;
    //@CompilerDirectives.CompilationFinal protected TruffleObject cusparseCreateCsrFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseCreateDnVecFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseCreateDnMatFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSpMV_bufferSizeFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSpVV_bufferSizeFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSpMM_bufferSizeFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSgemvi_bufferSizeFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseCgemvi_bufferSizeFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseDgemvi_bufferSizeFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseZgemvi_bufferSizeFunction;
    // spgemm
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSpGEMM_createDescrFunctionNFI;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSpGEMM_destroyDescrFunctionNFI;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSpGEMM_workEstimationFunctionNFI;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSpGEMM_computeFunctionNFI;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSpGEMM_copyFunctionNFI;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSpMatGetSizeFunctionNFI;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseCsrSetPointersFunctionNFI;


    private ExternalFunctionFactory externalFunctionFactory;
    protected Object[] args;
    private static GrCUDAContext context = null;

    public CUSPARSEProxy(ExternalFunctionFactory externalFunctionFactory) {
        this.externalFunctionFactory = externalFunctionFactory;
    }

    // we need to create a new context
    public static void setContext(GrCUDAContext context) {
        CUSPARSEProxy.context = context;
    }

    public static GrCUDAContext getContext() { return context; }

    protected void initializeNfi() {

        assert (context != null);

        String libraryPath = context.getOptions().getCuSPARSELibrary(); //getOption(GrCUDAOptions.CuSPARSELibrary);

        cusparseSetStreamFunctionNFI = CUSPARSE_CUSPARSESETSTREAM.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseCreateCooFunctionNFI = CUSPARSE_CUSPARSECREATECOO.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseCreateCsrFunctionNFI = CUSPARSE_CUSPARSECREATECSR.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseCreateDnVecFunctionNFI = CUSPARSE_CUSPARSECREATEDNVEC.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseCreateDnMatFunctionNFI = CUSPARSE_CUSPARSECREATEDNMAT.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseSpMV_bufferSizeFunctionNFI = CUSPARSE_CUSPARSESPMV_BUFFERSIZE.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseSpVV_bufferSizeFunctionNFI = CUSPARSE_CUSPARSESPVV_BUFFERSIZE.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseSpMM_bufferSizeFunctionNFI = CUSPARSE_CUSPARSESPMM_BUFFERSIZE.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseSgemvi_bufferSizeFunctionNFI = CUSPARSE_CUSPARSESGEMVI_BUFFERSIZE.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseCgemvi_bufferSizeFunctionNFI = CUSPARSE_CUSPARSECGEMVI_BUFFERSIZE.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseDgemvi_bufferSizeFunctionNFI = CUSPARSE_CUSPARSEDGEMVI_BUFFERSIZE.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseZgemvi_bufferSizeFunctionNFI = CUSPARSE_CUSPARSEZGEMVI_BUFFERSIZE.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseSpGEMM_createDescrFunctionNFI = CUSPARSE_SPGEMM_CREATEDESCR.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseSpGEMM_destroyDescrFunctionNFI = CUSPARSE_SPGEMM_DESTROYDESCR.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseSpGEMM_workEstimationFunctionNFI = CUSPARSE_SPGEMM_WORKESTIMATION.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseSpGEMM_computeFunctionNFI = CUSPARSE_SPGEMM_COMPUTE.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseSpGEMM_copyFunctionNFI = CUSPARSE_SPGEMM_COPY.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseSpMatGetSizeFunctionNFI = CUSPARSE_SPMATGETSIZE.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
        cusparseCsrSetPointersFunctionNFI = CUSPARSE_CSRSETPOINTERS.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);

        // cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr,
        // int64_t rows,
        // int64_t cols,
        // int64_t nnz,
        // void* cooRowInd,
        // void* cooColInd,
        // void* cooValues,
        // cusparseIndexType_t cooIdxType,
        // cusparseIndexBase_t idxBase,
        // cudaDataType valueType)

        cusparseCreateCooFunction = new Function(CUSPARSE_CUSPARSECREATECOO.getName()) {
            @Override
            @CompilerDirectives.TruffleBoundary
            public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                checkArgumentLength(arguments, 10);
                Long cusparseSpMatDescr = expectLong(arguments[0]);
                long rows = expectLong(arguments[1]);
                long cols = expectLong(arguments[2]);
                long nnz = expectLong(arguments[3]);
                DeviceArray cooRowIdx = (DeviceArray) arguments[4];
                DeviceArray cooColIdx = (DeviceArray) arguments[5];
                DeviceArray cooValues = (DeviceArray) arguments[6];
                CUSPARSERegistry.CUSPARSEIndexType cooIdxType = CUSPARSERegistry.CUSPARSEIndexType.values()[expectInt(arguments[7])];
                CUSPARSERegistry.CUSPARSEIndexBase cooIdxBase = CUSPARSERegistry.CUSPARSEIndexBase.values()[expectInt(arguments[8])];
                CUSPARSERegistry.CUDADataType valueType = CUSPARSERegistry.CUDADataType.values()[expectInt(arguments[9])];
                try {
                    Object result = INTEROP.execute(cusparseCreateCooFunctionNFI, cusparseSpMatDescr, rows, cols, nnz, cooRowIdx, cooColIdx, cooValues,
                                    cooIdxType.ordinal(), cooIdxBase.ordinal(), valueType.ordinal());
                    checkCUSPARSEReturnCode(result, "cusparseCreateCoo");
                    return result;
                } catch (InteropException e) {
                    throw new GrCUDAInternalException(e);
                }
            }
        };

        // cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr,
        // int64_t rows,
        // int64_t cols,
        // int64_t nnz,
        // void* csrRowOffsets,
        // void* csrColInd,
        // void* csrValues,
        // cusparseIndexType_t csrRowOffsetsType,
        // cusparseIndexType_t csrColIndType,
        // cusparseIndexBase_t idxBase,
        // cudaDataType valueType)
        /*cusparseCreateCsrFunction = new Function(CUSPARSE_CUSPARSECREATECSR.getName()) {
            @Override
            @CompilerDirectives.TruffleBoundary
            public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                checkArgumentLength(arguments, 11);
                Long cusparseSpMatDescr = expectLong(arguments[0]);
                long rows = expectLong(arguments[1]);
                long cols = expectLong(arguments[2]);
                long nnz = expectLong(arguments[3]);
                DeviceArray csrRowOffsets = (DeviceArray) arguments[4];
                DeviceArray csrColIdx = (DeviceArray) arguments[5];
                DeviceArray csrValues = (DeviceArray) arguments[6];
                CUSPARSERegistry.CUSPARSEIndexType csrRowOffsetsType = CUSPARSERegistry.CUSPARSEIndexType.values()[expectInt(arguments[7])];
                CUSPARSERegistry.CUSPARSEIndexType csrColIdxType = CUSPARSERegistry.CUSPARSEIndexType.values()[expectInt(arguments[8])];
                CUSPARSERegistry.CUSPARSEIndexBase csrIdxBase = CUSPARSERegistry.CUSPARSEIndexBase.values()[expectInt(arguments[9])];
                CUSPARSERegistry.CUDADataType valueType = CUSPARSERegistry.CUDADataType.values()[expectInt(arguments[10])];
                try {
                    Object result = INTEROP.execute(cusparseCreateCsrFunctionNFI, cusparseSpMatDescr, rows, cols, nnz, csrRowOffsets, csrColIdx, csrValues,
                                    csrRowOffsetsType.ordinal(), csrColIdxType.ordinal(), csrIdxBase.ordinal(), valueType.ordinal());
                    checkCUSPARSEReturnCode(result, "cusparseCreateCsr");
                    return result;
                } catch (InteropException e) {
                    throw new GrCUDAInternalException(e);
                }
            }
        };*/

        // cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
        // int64_t size,
        // void* values,
        // cudaDataType valueType)
        cusparseCreateDnVecFunction = new Function(CUSPARSE_CUSPARSECREATEDNVEC.getName()) {
            @Override
            @CompilerDirectives.TruffleBoundary
            public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                checkArgumentLength(arguments, 4);
                Long cusparseDnVecDescr = expectLong(arguments[0]);
                long size = expectLong(arguments[1]);
                DeviceArray values = (DeviceArray) arguments[2];
                CUSPARSERegistry.CUDADataType valueType = CUSPARSERegistry.CUDADataType.values()[expectInt(arguments[3])];
                try {
                    Object result = INTEROP.execute(cusparseCreateDnVecFunctionNFI, cusparseDnVecDescr, size, values, valueType.ordinal());
                    checkCUSPARSEReturnCode(result, "cusparseCreateDnVec");
                    return result;
                } catch (InteropException e) {
                    throw new GrCUDAInternalException(e);
                }
            }

        };

        // cusparseStatus_t
        //cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr,
        //                    int64_t               rows,
        //                    int64_t               cols,
        //                    int64_t               ld,
        //                    void*                 values,
        //                    cudaDataType          valueType,
        //                    cusparseOrder_t       order)
        cusparseCreateDnMatFunction = new Function(CUSPARSE_CUSPARSECREATEDNMAT.getName()) {
            @Override
            @CompilerDirectives.TruffleBoundary
            public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                checkArgumentLength(arguments, 7);
                Long cusparseDnMatDescr = expectLong(arguments[0]);
                long rows = expectLong(arguments[1]);
                long cols = expectLong(arguments[2]);
                long ld = expectLong(arguments[3]);
                DeviceArray values = (DeviceArray) arguments[4];
                CUSPARSERegistry.CUDADataType valueType = CUSPARSERegistry.CUDADataType.values()[expectInt(arguments[5])];
                CUSPARSERegistry.CUSPARSEOrder order = CUSPARSERegistry.CUSPARSEOrder.values()[expectInt(arguments[6])];
                try {
                    Object result = INTEROP.execute(cusparseCreateDnMatFunctionNFI, cusparseDnMatDescr, rows, cols, ld, values, valueType.ordinal(), order.ordinal());
                    checkCUSPARSEReturnCode(result, "cusparseCreateDnMat");
                    return result;
                } catch (InteropException e) {
                    throw new GrCUDAInternalException(e);
                }
            }

        };

        // cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle,
        // cusparseOperation_t opA,
        // const void* alpha,
        // cusparseSpMatDescr_t matA,
        // cusparseDnVecDescr_t vecX,
        // const void* beta,
        // cusparseDnVecDescr_t vecY,
        // cudaDataType computeType,
        // cusparseSpMVAlg_t alg,
        // size_t* bufferSize)
        cusparseSpMV_bufferSizeFunction = new Function(CUSPARSE_CUSPARSESPMV_BUFFERSIZE.getName()) {
            @Override
            @CompilerDirectives.TruffleBoundary
            public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                checkArgumentLength(arguments, 10);
                long handle = expectLong(arguments[0]);
                CUSPARSERegistry.CUSPARSEOperation opA = CUSPARSERegistry.CUSPARSEOperation.values()[expectInt(arguments[1])];
                DeviceArray alpha = (DeviceArray) arguments[2];
                long cusparseSpMatDesc = expectLong(arguments[3]);
                long vecX = expectLong(arguments[4]);
                DeviceArray beta = (DeviceArray) arguments[5];
                long vecY = expectLong(arguments[6]);
                CUSPARSERegistry.CUDADataType computeType = CUSPARSERegistry.CUDADataType.values()[expectInt(arguments[7])];
                CUSPARSERegistry.CUSPARSESpMVAlg alg = CUSPARSERegistry.CUSPARSESpMVAlg.values()[expectInt(arguments[8])];
                long bufferSize = expectLong(arguments[9]);
                try {
                    Object result = INTEROP.execute(cusparseSpMV_bufferSizeFunctionNFI, handle, opA.ordinal(), alpha, cusparseSpMatDesc, vecX, beta, vecY, computeType.ordinal(), alg.ordinal(),
                                    bufferSize);
                    checkCUSPARSEReturnCode(result, "cusparseSpMV_bufferSize");
                    return result;
                } catch (InteropException e) {
                    throw new GrCUDAInternalException(e);
                }
            }
        };


        cusparseSpVV_bufferSizeFunction = new Function(CUSPARSE_CUSPARSESPVV_BUFFERSIZE.getName()) {
            @Override
            @CompilerDirectives.TruffleBoundary
            public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                checkArgumentLength(arguments, 7);
                long handle = expectLong(arguments[0]);
                CUSPARSERegistry.CUSPARSEOperation opA = CUSPARSERegistry.CUSPARSEOperation.values()[expectInt(arguments[1])];
                long cusparseSpMatDesc = expectLong(arguments[2]);
                long cusparseDnVecDesc = expectLong(arguments[3]);
                DeviceArray r = (DeviceArray) arguments[4];
                CUSPARSERegistry.CUDADataType computeType = CUSPARSERegistry.CUDADataType.values()[expectInt(arguments[5])];
                long bufferSize = expectLong(arguments[6]);
                try {
                    Object result = INTEROP.execute(cusparseSpVV_bufferSizeFunctionNFI, handle, opA.ordinal(), cusparseSpMatDesc, cusparseDnVecDesc, r, computeType.ordinal(),
                            bufferSize);
                    checkCUSPARSEReturnCode(result, "cusparseSpVV_bufferSize");
                    return result;
                } catch (InteropException e) {
                    throw new GrCUDAInternalException(e);
                }
            }
        };

        // cusparseStatus_t
        //cusparseSpMM_bufferSize(cusparseHandle_t     handle,
        //                        cusparseOperation_t  opA,
        //                        cusparseOperation_t  opB,
        //                        const void*          alpha,
        //                        cusparseSpMatDescr_t matA,
        //                        cusparseDnMatDescr_t matB,
        //                        const void*          beta,
        //                        cusparseDnMatDescr_t matC,
        //                        cudaDataType         computeType,
        //                        cusparseSpMMAlg_t    alg,
        //                        size_t*              bufferSize)
        cusparseSpMM_bufferSizeFunction = new Function(CUSPARSE_CUSPARSESPMM_BUFFERSIZE.getName()) {
            @Override
            @CompilerDirectives.TruffleBoundary
            public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                checkArgumentLength(arguments, 11);
                long handle = expectLong(arguments[0]);
                CUSPARSERegistry.CUSPARSEOperation opA = CUSPARSERegistry.CUSPARSEOperation.values()[expectInt(arguments[1])];
                CUSPARSERegistry.CUSPARSEOperation opB = CUSPARSERegistry.CUSPARSEOperation.values()[expectInt(arguments[2])];
                DeviceArray alpha = (DeviceArray) arguments[3];
                long matA = expectLong(arguments[4]);
                long matB = expectLong(arguments[5]);
                DeviceArray beta = (DeviceArray) arguments[6];
                long matC = expectLong(arguments[7]);
                CUSPARSERegistry.CUDADataType computeType = CUSPARSERegistry.CUDADataType.values()[expectInt(arguments[8])];
                int alg = expectInt(arguments[9]);
                long bufferSize = expectLong(arguments[10]);
                try {
                    Object result = INTEROP.execute(cusparseSpVV_bufferSizeFunctionNFI, handle, opA.ordinal(),
                            opB.ordinal(), alpha, matA, matB, beta, matC, computeType.ordinal(), alg, bufferSize);
                    checkCUSPARSEReturnCode(result, "cusparseSpMM_bufferSize");
                    return result;
                } catch (InteropException e) {
                    throw new GrCUDAInternalException(e);
                }
            }
        };

        // cusparseStatus_t cusparseSgemvi_bufferSize(cusparseHandle_t handle,
        // cusparseOperation_t transA,
        // int m,
        // int n,
        // int nnz,
        // int* pBufferSize)
        cusparseSgemvi_bufferSizeFunction = new Function(CUSPARSE_CUSPARSESGEMVI_BUFFERSIZE.getName()) {
            @Override
            @CompilerDirectives.TruffleBoundary
            public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                checkArgumentLength(arguments, 6);
                long handle = expectLong(arguments[0]);
                CUSPARSERegistry.CUSPARSEOperation transA = CUSPARSERegistry.CUSPARSEOperation.values()[expectInt(arguments[1])];
                int rows = expectInt(arguments[2]);
                int cols = expectInt(arguments[3]);
                int nnz = expectInt(arguments[4]);
                long pBufferSize = expectLong(arguments[5]);
                try {
                    Object result = INTEROP.execute(cusparseSgemvi_bufferSizeFunctionNFI, handle, transA.ordinal(), rows, cols, nnz, pBufferSize);
                    checkCUSPARSEReturnCode(result, "cusparseSgemvi_bufferSize");
                    return result;
                } catch (InteropException e) {
                    throw new GrCUDAInternalException(e);
                }
            }
        };

        cusparseCgemvi_bufferSizeFunction = new Function(CUSPARSE_CUSPARSECGEMVI_BUFFERSIZE.getName()) {
            @Override
            @CompilerDirectives.TruffleBoundary
            public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                checkArgumentLength(arguments, 6);
                long handle = expectLong(arguments[0]);
                CUSPARSERegistry.CUSPARSEOperation transA = CUSPARSERegistry.CUSPARSEOperation.values()[expectInt(arguments[1])];
                int rows = expectInt(arguments[2]);
                int cols = expectInt(arguments[3]);
                int nnz = expectInt(arguments[4]);
                long pBufferSize = expectLong(arguments[5]);
                try {
                    Object result = INTEROP.execute(cusparseCgemvi_bufferSizeFunctionNFI, handle, transA.ordinal(), rows, cols, nnz, pBufferSize);
                    checkCUSPARSEReturnCode(result, "cusparseCgemvi_bufferSize");
                    return result;
                } catch (InteropException e) {
                    throw new GrCUDAInternalException(e);
                }
            }
        };

        cusparseDgemvi_bufferSizeFunction = new Function(CUSPARSE_CUSPARSEDGEMVI_BUFFERSIZE.getName()) {
            @Override
            @CompilerDirectives.TruffleBoundary
            public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                checkArgumentLength(arguments, 6);
                long handle = expectLong(arguments[0]);
                CUSPARSERegistry.CUSPARSEOperation transA = CUSPARSERegistry.CUSPARSEOperation.values()[expectInt(arguments[1])];
                int rows = expectInt(arguments[2]);
                int cols = expectInt(arguments[3]);
                int nnz = expectInt(arguments[4]);
                long pBufferSize = expectLong(arguments[5]);
                try {
                    Object result = INTEROP.execute(cusparseDgemvi_bufferSizeFunctionNFI, handle, transA.ordinal(), rows, cols, nnz, pBufferSize);
                    checkCUSPARSEReturnCode(result, "cusparseDgemvi_bufferSize");
                    return result;
                } catch (InteropException e) {
                    throw new GrCUDAInternalException(e);
                }
            }
        };

        cusparseZgemvi_bufferSizeFunction = new Function(CUSPARSE_CUSPARSEZGEMVI_BUFFERSIZE.getName()) {
            @Override
            @CompilerDirectives.TruffleBoundary
            public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                checkArgumentLength(arguments, 6);
                long handle = expectLong(arguments[0]);
                CUSPARSERegistry.CUSPARSEOperation transA = CUSPARSERegistry.CUSPARSEOperation.values()[expectInt(arguments[1])];
                int rows = expectInt(arguments[2]);
                int cols = expectInt(arguments[3]);
                int nnz = expectInt(arguments[4]);
                long pBufferSize = expectLong(arguments[5]);
                try {
                    Object result = INTEROP.execute(cusparseZgemvi_bufferSizeFunctionNFI, handle, transA.ordinal(), rows, cols, nnz, pBufferSize);
                    checkCUSPARSEReturnCode(result, "cusparseZgemvi_bufferSize");
                    return result;
                } catch (InteropException e) {
                    throw new GrCUDAInternalException(e);
                }
            }
        };


    }

    public ExternalFunctionFactory getExternalFunctionFactory() {
        return externalFunctionFactory;
    }

    public Object[] formatArguments(Object[] rawArgs, long handle) throws UnsupportedTypeException, UnsupportedMessageException, ArityException {
        this.initializeNfi();
        cudaDeviceSynchronize();
        return rawArgs;
    }

    public boolean requiresHandle() {
        return true;
    }

    @CompilerDirectives.TruffleBoundary
    public void cudaDeviceSynchronize() {
        try {
            Object callable = CUDARuntime.CUDARuntimeFunction.CUDA_DEVICESYNCHRONIZE.getSymbol(context.getCUDARuntime());
            Object result = INTEROP.execute(callable);
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }


    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESETSTREAM = new ExternalFunctionFactory("cusparseSetStream", "cusparseSetStream", "(sint64, sint64): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATECOO = new ExternalFunctionFactory("cusparseCreateCoo", "cusparseCreateCoo", "(pointer, sint64, " +
                    "sint64, sint64, pointer, pointer, pointer, sint32, sint32, sint32): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATECSR = new ExternalFunctionFactory("cusparseCreateCsr", "cusparseCreateCsr", "(pointer, sint64, sint64, sint64," +
                    "pointer, pointer, pointer, sint32, sint32, sint32, sint32): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATEDNVEC = new ExternalFunctionFactory("cusparseCreateDnVec", "cusparseCreateDnVec", "(pointer, sint64, pointer, " +
            "sint32): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATEDNMAT = new ExternalFunctionFactory("cusparseCreateDnMat", "cusparseCreateDnMat", "(pointer, sint64, sint64, " +
            "sint64, pointer, sint32, sint32): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESPMV_BUFFERSIZE = new ExternalFunctionFactory("cusparseSpMV_bufferSize", "cusparseSpMV_bufferSize", "(sint64, sint32," +
                    "pointer, sint64, sint64, pointer, sint64, sint32, sint32, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESPVV_BUFFERSIZE = new ExternalFunctionFactory("cusparseSpVV_bufferSize", "cusparseSpVV_bufferSize", "(sint64, sint32," +
            "sint64, sint64, pointer, sint32, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESPMM_BUFFERSIZE = new ExternalFunctionFactory("cusparseSpMM_bufferSize", "cusparseSpMM_bufferSize", "(sint64, sint32," +
            "sint32, pointer, sint64, sint64, pointer, sint64, sint32, sint32, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESGEMVI_BUFFERSIZE = new ExternalFunctionFactory("cusparseSgemvi_bufferSize", "cusparseSgemvi_bufferSize", "(sint64, sint32, " +
                    "sint64, sint64, sint64, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECGEMVI_BUFFERSIZE = new ExternalFunctionFactory("cusparseCgemvi_bufferSize", "cusparseCgemvi_bufferSize", "(sint64, sint32, " +
                    "sint64, sint64, sint64, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSEDGEMVI_BUFFERSIZE = new ExternalFunctionFactory("cusparseDgemvi_bufferSize", "cusparseDgemvi_bufferSize", "(sint64, sint32, " +
                    "sint64, sint64, sint64, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSEZGEMVI_BUFFERSIZE = new ExternalFunctionFactory("cusparseZgemvi_bufferSize", "cusparseZgemvi_bufferSize", "(sint64, sint32, " +
                    "sint64, sint64, sint64, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_SPGEMM_CREATEDESCR = new ExternalFunctionFactory("cusparseSpGEMM_createDescr", "cusparseSpGEMM_createDescr", "(sint64): sint32");
    private static final ExternalFunctionFactory CUSPARSE_SPGEMM_DESTROYDESCR = new ExternalFunctionFactory("cusparseSpGEMM_destroyDescr", "cusparseSpGEMM_destroyDescr", "(sint64): sint32");
    private static final ExternalFunctionFactory CUSPARSE_SPGEMM_WORKESTIMATION = new ExternalFunctionFactory("cusparseSpGEMM_workEstimation", "cusparseSpGEMM_workEstimation",
            "(sint64, sint32, sint32, pointer, sint64, sint64, pointer, sint64, sint32, sint32, sint64, sint64, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_SPGEMM_COMPUTE = new ExternalFunctionFactory("cusparseSpGEMM_compute", "cusparseSpGEMM_compute",
            "(sint64, sint32, sint32, pointer, sint64, sint64, pointer, sint64, sint32, sint32, sint64, sint64, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_SPGEMM_COPY = new ExternalFunctionFactory("cusparseSpGEMM_copy", "cusparseSpGEMM_copy",
            "(sint64, sint32, sint32, pointer, sint64, sint64, pointer, sint64, sint32, sint32, sint64): sint32");
    private static final ExternalFunctionFactory CUSPARSE_SPMATGETSIZE = new ExternalFunctionFactory("cusparseSpMatGetSize", "cusparseSpMatGetSize",
            "(sint64, sint64, sint64, sint64): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CSRSETPOINTERS = new ExternalFunctionFactory("cusparseCsrSetPointers", "cusparseCsrSetPointers",
            "(sint64, pointer, pointer, pointer): sint32");
    private static final ExternalFunctionFactory CUDA_DEVICESYNCHRONIZE = new ExternalFunctionFactory("cudaDeviceSynchronize","cudaDeviceSynchronize","(): sint32");
}
