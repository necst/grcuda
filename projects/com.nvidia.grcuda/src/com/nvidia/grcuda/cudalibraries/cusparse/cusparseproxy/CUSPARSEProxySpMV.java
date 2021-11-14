package com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Source;
import org.graalvm.polyglot.Value;
import static com.nvidia.grcuda.functions.Function.expectLong;
import static com.nvidia.grcuda.functions.Function.expectInt;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.io.OutputStream;
import java.util.logging.Handler;
// ci serve importare una classe dai test AHEM o ricreiamo tutto?

public class CUSPARSEProxySpMV extends CUSPARSEProxy {

    private final int nArgsRaw = 10; // args for library function

    public CUSPARSEProxySpMV(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs) throws UnsupportedTypeException {

        if(rawArgs.length == nArgsRaw){
            return rawArgs;
        } else {
            args = new Object[nArgsRaw];
            long rows = expectLong(rawArgs[3]);
            long cols = expectLong(rawArgs[4]);
            long nnz = expectLong(rawArgs[5]);

            if((rows == cols)&(cols == nnz)){ // coo
// cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
//                    int64_t               size,
//                    void*                 values,
//                    cudaDataType          valueType)
//            cusparseStatus_t
//            cusparseSpMV_bufferSize(cusparseHandle_t     handle,
//                    cusparseOperation_t  opA,
//                        const void*          alpha,
//                    cusparseSpMatDescr_t matA,
//                    cusparseDnVecDescr_t vecX,
//                        const void*          beta,
//                    cusparseDnVecDescr_t vecY,
//                    cudaDataType         computeType,
//                    cusparseSpMVAlg_t    alg,
//                    size_t*              bufferSize)
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
                // create context variable//bitwise non si può

                UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object cooMatDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();
                long handle = expectLong(rawArgs[0]);
                CUSPARSERegistry.cusparseOperation_t opA = CUSPARSERegistry.cusparseOperation_t.values()[expectInt(rawArgs[1])];
                long alpha = expectLong(rawArgs[2]);
//                rows = expectLong(rawArgs[3]);
//                cols = expectLong(rawArgs[3]);
//                nnz = expectLong(rawArgs[3]);
                long cooRowIdx = expectLong(rawArgs[6]);
                long cooColIdx = expectLong(rawArgs[7]);
                long cooValues = expectLong(rawArgs[8]);
                CUSPARSERegistry.cusparseIndexType_t cooIdxType = CUSPARSERegistry.cusparseIndexType_t.values()[expectInt(rawArgs[9])];
                CUSPARSERegistry.cusparseIndexBase_t cooIdxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[expectInt(rawArgs[10])];
                CUSPARSERegistry.cudaDataType valueType = CUSPARSERegistry.cudaDataType.values()[expectInt(rawArgs[11])];
                long size = cols; // check row
                long valuesX = expectLong(rawArgs[12]);
                CUSPARSERegistry.cudaDataType valueTypeVec = CUSPARSERegistry.cudaDataType.values()[expectInt(rawArgs[13])]; // same type everyone, for now (to avoid mismatches)
                long beta = expectLong(rawArgs[14]);
                long valuesY = expectLong(rawArgs[15]);
                CUSPARSERegistry.cusparseSpMVAlg_t alg = CUSPARSERegistry.cusparseSpMVAlg_t.values()[expectInt(rawArgs[16])];

                // create context for functions' execution
                Context polyglot = GrCUDAContext.buildProxyContext().build();
                Value cusparseCreateCoo = polyglot.eval("grcuda", "SPARSE::cusparseSpMV");

                // create coo matrix descriptor
                cusparseCreateCoo.execute(cooMatDescr.getAddress(), rows, cols, nnz, cooRowIdx, cooColIdx, cooValues, cooIdxType.ordinal(), cooIdxBase.ordinal(), valueType.ordinal()); // TODO: check enums

                // create dense vectors X and Y descriptors
                Value cusparseCreateDnVec = polyglot.eval("grcuda", "SPARSE::cusparseCreateDnVec");
                cusparseCreateDnVec.execute(dnVecXDescr.getAddress(), size, valuesX, valueTypeVec.ordinal());
                cusparseCreateDnVec.execute(dnVecYDescr.getAddress(), size, valuesY, valueTypeVec.ordinal());

                // create buffer
                Value cusparseSpMV_bufferSize = polyglot.eval("grcuda", "SPARSE::cusparseSpMV_bufferSize");
                cusparseSpMV_bufferSize.execute(handle, opA.ordinal(), alpha, cooMatDescr.getValue(), dnVecXDescr.getValue(), beta, dnVecYDescr.getValue(), valueType.ordinal(), alg.ordinal(), bufferSize.getAddress());

                // format new arguments for SpMV with COO format
                args[0] = handle;
                args[1] = opA;
                args[2] = alpha;
                args[3] = cooMatDescr;
                args[4] = dnVecXDescr;
                args[5] = beta;
                args[6] = dnVecYDescr;
                args[7] = valueType;
                args[8] = alg;
                args[9] = bufferSize;
            } else { // csr
                UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object csrMatDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();
// cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
//                    int64_t               size,
//                    void*                 values,
//                    cudaDataType          valueType)
//            cusparseStatus_t
//  cusparseSpMV_bufferSize(cusparseHandle_t     handle,
//                    cusparseOperation_t  opA,
//                        const void*          alpha,
//                    cusparseSpMatDescr_t matA,
//                    cusparseDnVecDescr_t vecX,
//                        const void*          beta,
//                    cusparseDnVecDescr_t vecY,
//                    cudaDataType         computeType,
//                    cusparseSpMVAlg_t    alg,
//                    size_t*              bufferSize)
//cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr,
//                  int64_t               rows,
//                  int64_t               cols,
//                  int64_t               nnz,
//                  void*                 csrRowOffsets,
//                  void*                 csrColInd,
//                  void*                 csrValues,
//                  cusparseIndexType_t   csrRowOffsetsType,
//                  cusparseIndexType_t   csrColIndType,
//                  cusparseIndexBase_t   idxBase,
//                  cudaDataType          valueType
                long handle = expectLong(rawArgs[0]);
                CUSPARSERegistry.cusparseOperation_t opA = CUSPARSERegistry.cusparseOperation_t.values()[expectInt(rawArgs[1])];
                long alpha = expectLong(rawArgs[2]);
//                rows = expectLong(rawArgs[0]);
//                cols = expectLong(rawArgs[0]);
//                nnz = expectLong(rawArgs[0]);
                long csrRowOffsets = expectLong(rawArgs[6]);
                long csrColIdx = expectLong(rawArgs[7]);
                long csrValues = expectLong(rawArgs[8]);
                CUSPARSERegistry.cusparseIndexType_t csrOffsetType = CUSPARSERegistry.cusparseIndexType_t.values()[expectInt(rawArgs[9])];
                CUSPARSERegistry.cusparseIndexType_t csrColIdxType = csrOffsetType; // all the same (for now)
                CUSPARSERegistry.cusparseIndexBase_t csrIdxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[expectInt(rawArgs[10])];
                CUSPARSERegistry.cudaDataType valueType = CUSPARSERegistry.cudaDataType.values()[expectInt(rawArgs[11])];
                long size = cols;
                long valuesX = expectLong(rawArgs[12]);
                CUSPARSERegistry.cudaDataType valueTypeVec = CUSPARSERegistry.cudaDataType.values()[expectInt(rawArgs[10])];
                long beta = expectLong(rawArgs[13]);
                long valuesY = expectLong(rawArgs[14]);
                CUSPARSERegistry.cusparseSpMVAlg_t alg = CUSPARSERegistry.cusparseSpMVAlg_t.values()[expectInt(rawArgs[15])];

                // create context:
                Context polyglot = GrCUDAContext.buildProxyContext().build();

                // create csr matrix descriptor
                Value cusparseCreateCsr = polyglot.eval("grcuda", "SPARSE::cusparseCreateCoo");
                cusparseCreateCsr.execute(csrMatDescr, rows, cols, nnz, csrRowOffsets, csrColIdx, csrValues, csrOffsetType, csrColIdxType, csrIdxBase, valueType); // TODO: check enums

                // create dense vectors X and Y descriptors
                Value cusparseCreateDnVec = polyglot.eval("grcuda", "SPARSE::cusparseCreateDnVec");
                cusparseCreateDnVec.execute(dnVecXDescr, size, valuesX, valueTypeVec);
                cusparseCreateDnVec.execute(dnVecYDescr, size, valuesY, valueTypeVec);

                // create buffer
                Value cusparseSpMV_bufferSize = polyglot.eval("grcuda", "SPARSE::cusparseSpMV_bufferSize");
                cusparseSpMV_bufferSize.execute(handle, opA, alpha, csrMatDescr, dnVecXDescr, beta, dnVecYDescr, valueType, alg, bufferSize);

                // format new arguments for SpMV with CSR format
                args[0] = handle;
                args[1] = opA;
                args[2] = alpha;
                args[3] = csrMatDescr;
                args[4] = dnVecXDescr;
                args[5] = beta;
                args[6] = dnVecYDescr;
                args[7] = valueType;
                args[8] = alg;
                args[9] = bufferSize;
            }
            return args;
        }
    }
    }
