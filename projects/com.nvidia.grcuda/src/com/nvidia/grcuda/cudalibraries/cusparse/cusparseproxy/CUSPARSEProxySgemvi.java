package com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy;

import org.graalvm.polyglot.Value;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;


public class CUSPARSEProxySgemvi extends CUSPARSEProxy {

    private final int nArgsRaw = 14; // args for library function
    private final int nArgsSimplified = 17; // args to be proxied


    public CUSPARSEProxySgemvi(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs) {
        if(rawArgs.length == nArgsRaw){
            return rawArgs;
        } else {

            // cusparseStatus_t cusparseSgemvi_bufferSize(cusparseHandle_t handle,
            //                  cusparseOperation_t transA,
            //                  int m,
            //                  int n,
            //                  int nnz,
            //                  int* pBufferSize)

//            cusparseStatus_t
//            cusparseSgemvi(cusparseHandle_t     handle,
//                    cusparseOperation_t  transA,
//            int                  m,
//            int                  n,
//               const float*         alpha,
//               const float*         A,
//            int                  lda,
//            int                  nnz,
//               const float*         x,
//               const int*           xInd,
//               const float*         beta,
//            float*               y,
//                    cusparseIndexBase_t  idxBase,
//            void*                pBuffer)

            args = new Object[nArgsRaw];
            UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();
            Value cu = polyglot.eval("grcuda", "CU");
            long handle = rawArgs[0];
            CUSPARSERegistry.cusparseOperation_t transA = CUSPARSERegistry.cusparseOperation_t.values()[rawArgs[1]];
            int m = rawArgs[2];
            int n = rawArgs[3];
            long alpha = rawArgs[4];
            long A = rawArgs[5];
            int lda = rawArgs[6];
            int nnz = rawArgs[7];
            long x = rawArgs[8];
            long xInd = rawArgs[9];
            long beta = rawArgs[10];
            long y = rawArgs[11];
            CUSPARSERegistry.cusparseIndexBase_t idxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[rawArgs[12]];
            Value cusparseSgemvi_bufferSize = polyglot.eval("grcuda", "SPARSE::cusparseSgemvi_bufferSize");
            cusparseSgemvi_bufferSize.execute(handle, transA, m, n, nnz, bufferSize);
            return args;
        }
    }
}
