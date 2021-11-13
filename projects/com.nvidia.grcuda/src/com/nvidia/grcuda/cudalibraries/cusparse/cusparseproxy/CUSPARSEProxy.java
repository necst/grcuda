package com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.functions.Function;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import static com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry.checkCUSPARSEReturnCode;

public abstract class CUSPARSEProxy {

    @CompilerDirectives.CompilationFinal
    private TruffleObject cusparseCreateFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseDestroyFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseSetStreamFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseCreateCooFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseCreateCsrFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseCreateDnVecFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseSpMV_bufferSizeFunctionNFI;
    @CompilerDirectives.CompilationFinal private TruffleObject cusparseSgemvi_bufferSizeFunctionNFI;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseCreateCooFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseCreateCsrFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseCreateDnVecFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSpMV_bufferSizeFunction;
    @CompilerDirectives.CompilationFinal protected TruffleObject cusparseSgemvi_bufferSizeFunction;

    private final int nArgsRaw = -1; // args for library function
    private final int nArgsSimplified = -1; // args to be proxied
    private ExternalFunctionFactory externalFunctionFactory;
    protected Object[] args;
    private static GrCUDAContext context = null;


    public CUSPARSEProxy(ExternalFunctionFactory externalFunctionFactory) {
        this.externalFunctionFactory = externalFunctionFactory;
        initializeNfi();
    }

    public static void setContext(GrCUDAContext context){
        CUSPARSEProxy.context = context;
    }

    private void initializeNfi(){
        if(context != null){
            String libraryPath = context.getOption(GrCUDAOptions.CuSPARSELibrary);

            cusparseSetStreamFunctionNFI = CUSPARSE_CUSPARSESETSTREAM.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
            cusparseCreateCooFunctionNFI = CUSPARSE_CUSPARSECREATECOO.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
            cusparseCreateCsrFunctionNFI = CUSPARSE_CUSPARSECREATECSR.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
            cusparseCreateDnVecFunctionNFI = CUSPARSE_CUSPARSECREATEDNVEC.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
            cusparseSpMV_bufferSizeFunctionNFI = CUSPARSE_CUSPARSESPMV_BUFFERSIZE.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);
            cusparseSgemvi_bufferSizeFunctionNFI = CUSPARSE_CUSPARSESGEMVI_BUFFERSIZE.makeFunction(context.getCUDARuntime(), libraryPath, CUSPARSERegistry.DEFAULT_LIBRARY_HINT);

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
                @CompilerDirectives.TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 10);
                    cusparseSpMatDescr = expectLong(arguments[0]);
                    long rows = expectLong(arguments[1]);
                    long cols = expectLong(arguments[2]);
                    long nnz = expectLong(arguments[3]);
                    long cooRowIdx = expectLong(arguments[4]);
                    long cooColIdx = expectLong(arguments[5]);
                    long cooValues = expectLong(arguments[6]);
                    CUSPARSERegistry.cusparseIndexType_t cooIdxType = CUSPARSERegistry.cusparseIndexType_t.values()[expectInt(arguments[7])];
                    CUSPARSERegistry.cusparseIndexBase_t cooIdxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[expectInt(arguments[8])];
                    CUSPARSERegistry.cudaDataType valueType = CUSPARSERegistry.cudaDataType.values()[expectInt(arguments[9])];
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
                @CompilerDirectives.TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 11);
                    cusparseSpMatDescr = expectLong(arguments[0]);
                    long rows = expectLong(arguments[1]);
                    long cols = expectLong(arguments[2]);
                    long nnz = expectLong(arguments[3]);
                    long csrRowOffsets = expectLong(arguments[4]);
                    long csrColIdx = expectLong(arguments[5]);
                    long csrValues = expectLong(arguments[6]);
                    CUSPARSERegistry.cusparseIndexType_t csrRowOffsetsType = CUSPARSERegistry.cusparseIndexType_t.values()[expectInt(arguments[7])];
                    CUSPARSERegistry.cusparseIndexType_t csrColIdxType = CUSPARSERegistry.cusparseIndexType_t.values()[expectInt(arguments[8])];
                    CUSPARSERegistry.cusparseIndexBase_t csrIdxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[expectInt(arguments[9])];
                    CUSPARSERegistry.cudaDataType valueType = CUSPARSERegistry.cudaDataType.values()[expectInt(arguments[10])];
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
                @CompilerDirectives.TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 4);
                    cusparseDnVecDescr = expectLong(arguments[0]);
                    long size = expectLong(arguments[1]);
                    long values = expectLong(arguments[2]);
                    CUSPARSERegistry.cudaDataType valueType = CUSPARSERegistry.cudaDataType.values()[expectInt(arguments[3])];
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
                @CompilerDirectives.TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 10);
                    long handle = expectLong(arguments[0]);
                    CUSPARSERegistry.cusparseOperation_t opA = CUSPARSERegistry.cusparseOperation_t.values()[expectInt(arguments[1])];
                    long alpha = expectLong(arguments[2]);
                    long cusparseSpMatDesc = expectLong(arguments[3]);
                    long vecX = expectLong(arguments[4]);
                    long beta = expectLong(arguments[5]);
                    long vecY = expectLong(arguments[6]);
                    CUSPARSERegistry.cudaDataType computeType = CUSPARSERegistry.cudaDataType.values()[expectInt(arguments[7])];
                    CUSPARSERegistry.cusparseSpMVAlg_t alg = CUSPARSERegistry.cusparseSpMVAlg_t.values()[expectInt(arguments[8])];
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
                @CompilerDirectives.TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 6);
                    long handle = expectLong(arguments[0]);
                    CUSPARSERegistry.cusparseOperation_t transA = CUSPARSERegistry.cusparseOperation_t.values()[expectInt(arguments[1])];
                    int m = expectInt(arguments[2]);
                    int n = expectInt(arguments[3]);
                    int nnz = expectInt(arguments[4]);
                    long pBufferSize = expectLong(arguments[5]);
                    try {
                        Object result = INTEROP.execute(cusparseSgemvi_bufferSizeFunctionNFI, handle, transA, m, n, nnz, pBufferSize);
                        checkCUSPARSEReturnCode(result, "cusparseSgemvi_bufferSize");
                        return result;
                    } catch (InteropException e){
                        throw new GrCUDAInternalException(e);
                    }
                }
            };


        } else {
            // TODO: FIGURE OUT WHAT TO THROW IN CASE CONTEXT IS NULL
        }

    }



    public ExternalFunctionFactory getExternalFunctionFactory() {
        return externalFunctionFactory;
    }

    public abstract Object[] formatArguments(Object[] rawArgs) throws UnsupportedTypeException;

    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESETSTREAM = new ExternalFunctionFactory("cusparseSetStream", "cusparseSetStream", "(sint64, sint64): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATECOO = new ExternalFunctionFactory("cusparseCreateCoo", "cusparseCreateCoo", "(pointer, sint64, " +
            "sint64, sint64, pointer, pointer, pointer, sint32, sint32, sint32): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATECSR = new ExternalFunctionFactory("cusparseCreateCsr", "cusparseCreateCsr", "(pointer, sint64, sint64, sint64," +
            "pointer, pointer, pointer, sint32, sint32, sint32, sint32): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATEDNVEC = new ExternalFunctionFactory("cusparseCreateDnVec", "cusparseCreateDnVec", "(pointer, sint64, pointer, " +
            "sint32): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESPMV_BUFFERSIZE = new ExternalFunctionFactory("cusparseSpMV_bufferSize", "cusparseSpMV_bufferSize", "(sint64, sint32," +
            "pointer, sint64, sint64, pointer, sint64, sint32, sint32, pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESGEMVI_BUFFERSIZE = new ExternalFunctionFactory("cusparseSpGEMM", "cusparseSpGEMM", "(sint64, sint32, sint32, " +
            "sint32, sint32, pointer): sint32");

}
