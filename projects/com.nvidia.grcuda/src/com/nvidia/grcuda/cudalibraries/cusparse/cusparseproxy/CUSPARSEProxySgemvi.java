package com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Source;
import org.graalvm.polyglot.Value;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectLong;
import static com.nvidia.grcuda.functions.Function.expectInt;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CUSPARSEProxySgemvi extends CUSPARSEProxy {

    private final int nArgsRaw = 14; // args for library function

    public CUSPARSEProxySgemvi(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs, long handle) throws UnsupportedTypeException {
        this.initializeNfi();
        if(rawArgs.length == nArgsRaw){
            return rawArgs;
        } else {
            args = new Object[nArgsRaw];
            UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();
            CUSPARSERegistry.cusparseOperation_t transA = CUSPARSERegistry.cusparseOperation_t.values()[expectInt(rawArgs[0])];
            int m = expectInt(rawArgs[1]);
            int n = expectInt(rawArgs[2]);
            DeviceArray alpha = (DeviceArray) rawArgs[3];
            long A = expectLong(rawArgs[0]);
            int lda = expectInt(rawArgs[0]);
            int nnz = expectInt(rawArgs[6]);
            long x = expectLong(rawArgs[0]);
            long xInd = expectLong(rawArgs[0]);
            long beta = expectLong(rawArgs[0]);
            long y = expectLong(rawArgs[0]);
            CUSPARSERegistry.cusparseIndexBase_t idxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[expectInt(rawArgs[0])];

            // create buffer
            try {
                Object resultBufferSize = INTEROP.execute(cusparseSgemvi_bufferSizeFunction, handle, transA.ordinal(), m, n, nnz, bufferSize.getAddress());
            } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
                e.printStackTrace();
            }

            long numElements;

            if (bufferSize.getValue() == 0) {
                numElements = 1;
            } else {
                numElements = (long) bufferSize.getValue();
            }

            DeviceArray buffer = new DeviceArray(alpha.getGrCUDAExecutionContext(), numElements, alpha.getElementType());

            args[0] = handle;
            args[1] = transA.ordinal();
            args[2] = m;
            args[3] = n;
            args[4] = alpha;
            args[5] = A;
            args[6] = lda;
            args[7] = nnz;
            args[8] = x;
            args[9] = xInd;
            args[10] = beta;
            args[11] = y;
            args[12] = idxBase;
            args[13] = buffer;

            return args;
        }
    }
}
