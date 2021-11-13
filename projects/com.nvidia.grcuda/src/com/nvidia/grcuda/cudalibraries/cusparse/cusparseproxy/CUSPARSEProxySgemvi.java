package com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy;
import org.graalvm.polyglot.Source;
import org.graalvm.polyglot.Value;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import static com.nvidia.grcuda.functions.Function.expectLong;
import static com.nvidia.grcuda.functions.Function.expectInt;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CUSPARSEProxySgemvi extends CUSPARSEProxy {

    private final int nArgsRaw = 14; // args for library function

    public CUSPARSEProxySgemvi(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Value eval(Source source) {
        return null;
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs) throws UnsupportedTypeException {
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
            long handle = expectLong(rawArgs[0]);
            CUSPARSERegistry.cusparseOperation_t transA = CUSPARSERegistry.cusparseOperation_t.values()[expectInt(rawArgs[0])];
            int m = expectInt(rawArgs[0]);
            int n = expectInt(rawArgs[0]);
//            long alpha = expectLong(rawArgs[0]);
//            long A = expectLong(rawArgs[0]);
//            int lda = expectInt(rawArgs[0]);
            int nnz = expectInt(rawArgs[0]);
//            long x = expectLong(rawArgs[0]);
//            long xInd = expectLong(rawArgs[0]);
//            long beta = expectLong(rawArgs[0]);
//            long y = expectLong(rawArgs[0]);
//            CUSPARSERegistry.cusparseIndexBase_t idxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[expectInt(rawArgs[0])];

            // create buffer
            Value cusparseSgemvi_bufferSize = eval(Source.create("grcuda", "SPARSE::cusparseSgemvi_bufferSize"));
            cusparseSgemvi_bufferSize.execute(handle, transA.ordinal(), m, n, nnz, bufferSize.getAddress());
            return args;
        }
    }
}
