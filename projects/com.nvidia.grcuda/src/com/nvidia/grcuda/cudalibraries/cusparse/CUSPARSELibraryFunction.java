/*
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

import java.util.ArrayList;
import java.util.List;
import com.nvidia.grcuda.functions.Function;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.cudalibraries.CUDALibraryFunction;
import com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy.CUSPARSEProxy;
import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.runtime.computation.FunctionExecution;
import com.oracle.truffle.api.CompilerDirectives;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import static com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry.DEFAULT_LIBRARY_HINT;
import com.nvidia.grcuda.runtime.stream.LibrarySetStream;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

/**
 * Wrapper class to CUDA library functions. It holds the signature of the function being wrapped,
 * and creates {@link ComputationArgument} for the signature and inputs;
 */
public class CUSPARSELibraryFunction extends CUDALibraryFunction {
    private CUSPARSERegistry registry;
    private Function nfiFunction;
    private final CUSPARSEProxy proxy;
    private final Long cusparseHandle;
    private final LibrarySetStream cusparseLibrarySetStream;
    private final GrCUDAContext context;
    private final String libraryPath;

    public CUSPARSELibraryFunction(CUSPARSERegistry registry, CUSPARSEProxy proxy, Long cusparseHandle, LibrarySetStream cusparseLibrarySetStream, GrCUDAContext context, String libraryPath) {
        super(proxy.getExternalFunctionFactory().getName(), proxy.getExternalFunctionFactory().getNFISignature());
        this.registry = registry;
        this.proxy = proxy;
        this.cusparseHandle = cusparseHandle;
        this.cusparseLibrarySetStream = cusparseLibrarySetStream;
        this.context = context;
        this.libraryPath = libraryPath;
    }


    @Override
    public List<ComputationArgumentWithValue> createComputationArgumentWithValueList(Object[] args, Long libraryHandle) {
        final int nInputArgs = this.computationArguments.size() - (libraryHandle == null ? 0 : 1);
        List<ComputationArgumentWithValue> argumentsWithValue = new ArrayList<>();

        int i = 0;
        if (libraryHandle != null) {
            argumentsWithValue.add(new ComputationArgumentWithValue(this.computationArguments.get(i++), libraryHandle));
        }

        // Set the other arguments;
        int j;
        for (j = 0; j < nInputArgs; ++j) {
            argumentsWithValue.add(new ComputationArgumentWithValue(this.computationArguments.get(i++), args[j]));
        }
        // Add extra arguments at the end: they are used to track input DeviceArrays;
        int numExtraArrays = args.length - nInputArgs;
        for (int k = 0; k < numExtraArrays; k++) {
            argumentsWithValue.add((ComputationArgumentWithValue) args[j++]);
        }
        return argumentsWithValue;
    }

    @Override
    @TruffleBoundary
    protected Object call(Object[] arguments) {
        registry.ensureInitialized();

        try {
            if (nfiFunction == null) {
                CompilerDirectives.transferToInterpreterAndInvalidate();
                nfiFunction = proxy.getExternalFunctionFactory().makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            }
            // This list of arguments might have extra arguments: the DeviceArrays that can cause dependencies but are not directly used by the cuSPARSE function,
            //   as these DeviceArrays might be wrapped using cuSparseMatrices/Vectors/Buffers.
            // We still need to pass these DeviceArrays to the GrCUDAComputationalElement so we track dependencies correctly,
            // but they are removed from the final list of arguments passed to the cuSPARSE library;
            Object[] formattedArguments = proxy.formatArguments(arguments, cusparseHandle);
            List<ComputationArgumentWithValue> computationArgumentsWithValue;
            if (proxy.requiresHandle()) {
                computationArgumentsWithValue = this.createComputationArgumentWithValueList(formattedArguments, cusparseHandle);
            } else {
                computationArgumentsWithValue = this.createComputationArgumentWithValueList(formattedArguments, null);
            }
            int extraArraysToTrack = computationArgumentsWithValue.size() - this.computationArguments.size();  // Both lists also contain the handle;
            Function fun = new Function(getName()) {
                @Override
                protected Object call(Object[] arguments) throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
                    arguments = proxy.executePreliminaries(arguments);
                    return INTEROP.execute(nfiFunction, arguments);
                }
            };
            Object result = new FunctionExecution(context.getGrCUDAExecutionContext(), fun, cusparseLibrarySetStream,
                    computationArgumentsWithValue, extraArraysToTrack).schedule();

            CUSPARSERegistry.checkCUSPARSEReturnCode(result, nfiFunction.getName());
            return result;
        } catch (InteropException e) {
            throw new GrCUDAInternalException(e);
        }
    }
}

