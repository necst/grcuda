package com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;

import java.util.Arrays;
import java.util.Collection;

import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;


public class CUSPARSEProxySpMV extends CUSPARSEProxy {

    private final int nArgsRaw = 10; // args for library function
    private final int nArgsSimplified = 17; // args to be proxied


    public CUSPARSEProxySpMV(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs) {
        if(rawArgs.length == nArgsRaw){
            return rawArgs;
        } else { // magari else if rawArgs.len = 17? just to be super safe
            // call functions to create arguments and descriptors
            args = new Object[nArgsRaw];
            if(rawArgs[1] && rawArgs[2] && rawArgs[3]){ // coo
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
                // create context variable
                Value cu = polyglot.eval("grcuda", "CU");
                UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object cooMatDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();
                // for clarity I rewrote all args, can be deleted
                long handle = rawArgs[0];
                CUSPARSERegistry.cusparseOperation_t opA = CUSPARSERegistry.cusparseOperation_t.values()[rawArgs[1]];
                long alpha = rawArgs[2];
                long rows = rawArgs[3];
                long cols = rawArgs[4];
                long nnz = rawArgs[5];
                long cooRowIdx = rawArgs[6];
                long cooColIdx = rawArgs[7];
                long cooValues = rawArgs[8];
                CUSPARSERegistry.cusparseIndexType_t cooIdxType = CUSPARSERegistry.cusparseIndexType_t.values()[rawArgs[9]];
                CUSPARSERegistry.cusparseIndexBase_t cooIdxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[rawArgs[10]];
                CUSPARSERegistry.cudaDataType valueType = CUSPARSERegistry.cudaDataType.values()[rawArgs[11]];
                long size = cols; // giusto? per moltiplicazioni riga per colonna dovrebbe, ma ricontrolliamo
                long valuesX = rawArgs[12];
                CUSPARSERegistry.cudaDataType valueTypeVec = CUSPARSERegistry.cudaDataType.values()[rawArgs[13]]; // per ora tutti i vettori sono dello stesso tipo
                long beta = rawArgs[14];
                long valuesY = rawArgs[15];
                CUSPARSERegistry.cusparseSpMVAlg_t alg = CUSPARSERegistry.cudaDataType.values()[rawArgs[16]];
                // now I can create a coo matrix:
                Value cusparseCreateCoo = polyglot.eval("grcuda", "SPARSE::cusparseCreateCoo");
                cusparseCreateCoo.execute(cooMatDescr, rows, cols, nnz, cooRowIdx, cooColIdx, cooValues, cooIdxType, cooIdxBase, valueType); // TODO: check enums
                // create dense vectors X and Y
                Value cusparseCreateDnVec = polyglot.eval("grcuda", "SPARSE::cusparseCreateDnVec");
                cusparseCreateDnVec.execute(dnVecXDescr, size, valuesX, valueTypeVec);
                cusparseCreateDnVec.execute(dnVecYDescr, size, valuesY, valueTypeVec);
                // create buffer
                Value cusparseSpMV_bufferSize = polyglot.eval("grcuda", "SPARSE::cusparseSpMV_bufferSize");
                cusparseSpMV_bufferSize.execute(handle, opA, alpha, cooMatDescr, dnVecXDescr, beta, dnVecYDescr, valueType, alg, bufferSize);

                // creating new arguments for SpMV with COO format
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
                Value cu = polyglot.eval("grcuda", "CU");
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
                long handle = rawArgs[0];
                CUSPARSERegistry.cusparseOperation_t opA = CUSPARSERegistry.cusparseOperation_t.values()[rawArgs[1]];
                long alpha = rawArgs[2];
                long rows = rawArgs[3];
                long cols = rawArgs[4];
                long nnz = rawArgs[5];
                long csrRowOffsets = rawArgs[6];
                long csrColIdx = rawArgs[7];
                long csrValues = rawArgs[8];
                CUSPARSERegistry.cusparseIndexType_t csrOffsetType = CUSPARSERegistry.cusparseIndexType_t.values()[rawArgs[9]];
                CUSPARSERegistry.cusparseIndexType_t csrColIdxType = csrOffsetType; // per ora uguali, poi vediamo come sistemare tutti gli argomenti
                CUSPARSERegistry.cusparseIndexBase_t csrIdxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[rawArgs[10]];
                CUSPARSERegistry.cudaDataType valueType = CUSPARSERegistry.cudaDataType.values()[rawArgs[11]];
                long size = cols; // giusto? per moltiplicazioni riga per colonna dovrebbe, ma ricontrolliamo
                long valuesX = rawArgs[12];
                CUSPARSERegistry.cudaDataType valueTypeVec = CUSPARSERegistry.cudaDataType.values()[rawArgs[13]]; // per ora tutti i vettori sono dello stesso tipo
                long beta = rawArgs[14];
                long valuesY = rawArgs[15];
                CUSPARSERegistry.cusparseSpMVAlg_t alg = CUSPARSERegistry.cudaDataType.values()[rawArgs[16]];
                // now I can create a csr matrix:
                Value cusparseCreateCsr = polyglot.eval("grcuda", "SPARSE::cusparseCreateCoo");
                cusparseCreateCsr.execute(csrMatDescr, rows, cols, nnz, csrRowOffsets, csrColIdx, csrValues, csrOffsetType, csrColIdxType, csrIdxBase, valueType); // TODO: check enums
                // create dense vectors X and Y
                Value cusparseCreateDnVec = polyglot.eval("grcuda", "SPARSE::cusparseCreateDnVec");
                cusparseCreateDnVec.execute(dnVecXDescr, size, valuesX, valueTypeVec);
                cusparseCreateDnVec.execute(dnVecYDescr, size, valuesY, valueTypeVec);
                // create buffer
                Value cusparseSpMV_bufferSize = polyglot.eval("grcuda", "SPARSE::cusparseSpMV_bufferSize");
                cusparseSpMV_bufferSize.execute(handle, opA, alpha, csrMatDescr, dnVecXDescr, beta, dnVecYDescr, valueType, alg, bufferSize);

                // creating new arguments for SpMV with COO format
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
