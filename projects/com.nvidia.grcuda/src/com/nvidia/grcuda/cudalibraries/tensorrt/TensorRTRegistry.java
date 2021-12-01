/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
package com.nvidia.grcuda.cudalibraries.tensorrt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;

import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.cudalibraries.CUDALibraryFunction;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.GPUPointer;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.Namespace;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.runtime.computation.CUDALibraryExecution;
import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.runtime.stream.LibrarySetStreamFunction;
import com.nvidia.grcuda.runtime.stream.TensorRTSetStreamFunction;
import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.CompilationFinal;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectLong;

public class TensorRTRegistry {

    private static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    public static final String DEFAULT_LIBRARY = "libtrt.so";
    public static final String NAMESPACE = "TRT";
    public static final String DEFAULT_LIBRARY_HINT = " (TensorRT library location can be set via the --grcuda.TensorRTLibrary= option. " +
                    "TensorRT support can be disabled via --grcuda.TensorRTEnabled=false.";

    private final GrCUDAContext context;
    private final String libraryPath;

    private LibrarySetStreamFunction tensorRTLibrarySetStreamFunction;

    @CompilationFinal private TruffleObject tensorRTCreateInferRuntime;
    @CompilationFinal private TruffleObject tensorRTDeserializeCudaEngine;
    @CompilationFinal private TruffleObject tensorRTDestroyInferRuntime;
    @CompilationFinal private TruffleObject tensorRTCreateExecutionContext;
    @CompilationFinal private TruffleObject tensorRTGetBindingIndexes;
    @CompilationFinal private TruffleObject tensorRTGetMaxBatchSize;
    @CompilationFinal private TruffleObject tensorRTEnqueue;
    @CompilationFinal private TruffleObject tensorRTDestroyEngine;
    @CompilationFinal private TruffleObject tensorRTDestroyExecutionContext;
    @CompilationFinal private TruffleObject tensorRTCreateInferRuntimeNFI;
    @CompilationFinal private TruffleObject tensorRTDeserializeCudaEngineNFI;
    @CompilationFinal private TruffleObject tensorRTDestroyInferRuntimeNFI;
    @CompilationFinal private TruffleObject tensorRTCreateExecutionContextNFI;
    @CompilationFinal private TruffleObject tensorRTGetBindingIndexesNFI;
    @CompilationFinal private TruffleObject tensorRTGetMaxBatchSizeNFI;
    @CompilationFinal private TruffleObject tensorRTEnqueueNFI;
    @CompilationFinal private TruffleObject tensorRTDestroyEngineNFI;
    @CompilationFinal private TruffleObject tensorRTDestroyExecutionContextNFI;

    private Integer runtime = null;

    public TensorRTRegistry(GrCUDAContext context) {
        this.context = context;
        libraryPath = context.getOption(GrCUDAOptions.TensorRTLibrary);
        context.addDisposable(this::shutdown);
    }

    public void ensureInitialized() {
        if (runtime == null) {
            CompilerDirectives.transferToInterpreterAndInvalidate();

            tensorRTCreateInferRuntimeNFI = TensorRTFunctionNFI.TRT_CREATE_INFER_RUNTIME.factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            tensorRTDeserializeCudaEngineNFI = TensorRTFunctionNFI.TRT_DESERIALIZE_CUDA_ENGINE.factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            tensorRTDestroyInferRuntimeNFI = TensorRTFunctionNFI.TRT_DESTROY_INFER_RUNTIME.factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            tensorRTCreateExecutionContextNFI = TensorRTFunctionNFI.TRT_CREATE_EXECUTION_CONTEXT.factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            tensorRTGetBindingIndexesNFI = TensorRTFunctionNFI.TRT_GET_BINDING_INDEX.factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            tensorRTGetMaxBatchSizeNFI = TensorRTFunctionNFI.TRT_GET_MAX_BATCH_SIZE.factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            tensorRTEnqueueNFI = TensorRTFunctionNFI.TRT_ENQUEUE.factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            tensorRTDestroyExecutionContextNFI = TensorRTFunctionNFI.TRT_DESTROY_EXECUTION_CONTEXT.factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);

            tensorRTCreateInferRuntime = new Function(TensorRTFunctionNFI.TRT_CREATE_INFER_RUNTIME.factory.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException {
                    checkArgumentLength(arguments, 0);
                    try {
                        Object result = INTEROP.execute(tensorRTCreateInferRuntimeNFI);
                        checkTRTReturnCode(result, "createInferRuntime");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            tensorRTDeserializeCudaEngine = new Function(TensorRTFunctionNFI.TRT_DESERIALIZE_CUDA_ENGINE.factory.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 2);
                    int param1 = expectInt(arguments[0]);
                    String param2 = expectString(arguments[1],"wrong parameter");
                    try {
                        Object result = INTEROP.execute(tensorRTDeserializeCudaEngineNFI, param1, param2);
                        checkTRTReturnCode(result, "deserializeCudaEngine");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            tensorRTDestroyInferRuntime = new Function(TensorRTFunctionNFI.TRT_DESTROY_INFER_RUNTIME.factory.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 1);
                    int param1 = expectInt(arguments[0]);
                    try {
                        Object result = INTEROP.execute(tensorRTDestroyInferRuntimeNFI, param1);
                        checkTRTReturnCode(result, "destroyInferRuntime");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            tensorRTCreateExecutionContext = new Function(TensorRTFunctionNFI.TRT_CREATE_EXECUTION_CONTEXT.factory.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 1);
                    int param1 = expectInt(arguments[0]);
                    try {
                        Object result = INTEROP.execute(tensorRTCreateExecutionContextNFI, param1);
                        checkTRTReturnCode(result, "createExecutionContext");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };


            tensorRTGetBindingIndexes = new Function(TensorRTFunctionNFI.TRT_GET_BINDING_INDEX.factory.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 2);
                    int param1 = expectInt(arguments[0]);
                    String param2 = expectString(arguments[1],"wrong paramter");
                    try {
                        Object result = INTEROP.execute(tensorRTGetBindingIndexesNFI, param1, param2);
                        checkTRTReturnCode(result, "getBindingIndex");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            tensorRTGetMaxBatchSize = new Function(TensorRTFunctionNFI.TRT_GET_MAX_BATCH_SIZE.factory.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 1);
                    int param1 = expectInt(arguments[0]);
                    try {
                        Object result = INTEROP.execute(tensorRTGetMaxBatchSizeNFI, param1);
                        checkTRTReturnCode(result, "getMaxBatchSize");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            tensorRTEnqueue = new Function(TensorRTFunctionNFI.TRT_ENQUEUE.factory.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 5);
                    int param1 = expectInt(arguments[0]);
                    int param2 = expectInt(arguments[1]);
                    long param3 = expectLong(arguments[2]);
                    long param4 = expectLong(arguments[3]);
                    long param5 = expectLong(arguments[4]);
                    try {
                        Object result = INTEROP.execute(tensorRTEnqueueNFI, param1, param2, param3, param4, param5);
                        checkTRTReturnCode(result, "enqueue");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            tensorRTDestroyEngine = new Function(TensorRTFunctionNFI.TRT_DESTROY_ENGINE.factory.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 1);
                    int param1 = expectInt(arguments[0]);
                    try {
                        Object result = INTEROP.execute(tensorRTDestroyEngineNFI, param1);
                        checkTRTReturnCode(result, "destroyEngine");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            tensorRTDestroyExecutionContext = new Function(TensorRTFunctionNFI.TRT_DESTROY_EXECUTION_CONTEXT.factory.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 1);
                    int param1 = expectInt(arguments[0]);
                    try {
                        Object result = INTEROP.execute(tensorRTDestroyExecutionContextNFI, param1);
                        checkTRTReturnCode(result, "destroyExecutionContext");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            try {
                Object result = INTEROP.execute(tensorRTCreateInferRuntime);
                runtime = INTEROP.asInt(result);

                context.addDisposable(this::tensorRTShutdown);
            } catch (InteropException e) {
                throw new GrCUDAInternalException(e);
            }
        }

        tensorRTLibrarySetStreamFunction = new TensorRTSetStreamFunction((Function) tensorRTEnqueueNFI);

    }

    private void tensorRTShutdown() {
        CompilerAsserts.neverPartOfCompilation();
        if (runtime != null) {
            try {
                Object result = InteropLibrary.getFactory().getUncached().execute(tensorRTDestroyInferRuntime);
                checkTRTReturnCode(result, TensorRTFunctionNFI.TRT_DESTROY_INFER_RUNTIME.factory.getName());
                runtime = null;
            } catch (InteropException e) {
                throw new GrCUDAInternalException(e);
            }
        }
    }

    public void registerTensorRTFunctions(Namespace namespace) {
        List<String> hiddenFunctions = Arrays.asList("enqueue");
        EnumSet.allOf(TensorRTFunctionNFI.class).stream().filter(func -> !hiddenFunctions.contains(func.getFunctionFactory().getName())).forEach(func -> {
            final ExternalFunctionFactory factory = func.getFunctionFactory();
            Function function = (func.checkError) ? new ErrorCheckedTRTFunction(factory) : new TRTFunction(factory);
            namespace.addFunction(function);
        });
        namespace.addFunction(new EnqueueFunction(TensorRTFunctionNFI.TRT_ENQUEUE.factory));
    }

    private void shutdown() {

    }

    private static void checkTRTReturnCode(Object result, String function) {
        int returnCode;
        try {
            returnCode = INTEROP.asInt(result);
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw new RuntimeException("expected return code as Integer object in " + function + ", got " + result.getClass().getName());
        }
        if (returnCode < 0) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(returnCode, trtReturnCodeToString(returnCode), new String[]{function});
        }
    }

    private static String trtReturnCodeToString(int returnCode) {
        switch (returnCode) {
            case 0:
                return "TRT_OK";
            case -1:
                return "TRT_INVALID_HANDLE";
            case -2:
                return "TRT_UNABLE_TO_CREATE_RUNTIME";
            case -3:
                return "TRT_ENGINE_DESERIALIZATION_ERROR";
            case -4:
                return "TRT_ENGINE_FILE_NOT_FOUND";
            default:
                return "unknown error code: " + returnCode;
        }
    }

    public enum TensorRTFunctionNFI {
        TRT_CREATE_INFER_RUNTIME(
                        new ExternalFunctionFactory("createInferRuntime", "createInferRuntime", "(): sint32"),
                        true),
        TRT_DESERIALIZE_CUDA_ENGINE(
                        new ExternalFunctionFactory("deserializeCudaEngine", "deserializeCudaEngine", "(sint32, string): sint32"),
                        true),
        TRT_DESTROY_INFER_RUNTIME(
                        new ExternalFunctionFactory("destroyInferRuntime", "destroyInferRuntime", "(sint32): sint32"),
                        true),
        TRT_CREATE_EXECUTION_CONTEXT(
                        new ExternalFunctionFactory("createExecutionContext", "createExecutionContext", "(sint32): sint32"),
                        true),
        TRT_GET_BINDING_INDEX(
                            new ExternalFunctionFactory("getBindingIndex", "getBindingIndex", "(sint32, string): sint32"),
                        false),
        TRT_GET_MAX_BATCH_SIZE(
                        new ExternalFunctionFactory("getMaxBatchSize", "getMaxBatchSize", "(sint32): sint32"),
                        false),
        TRT_ENQUEUE(
                        new ExternalFunctionFactory("enqueue", "enqueue", "(sint32, pointer, pointer, pointer): sint32"),
                        false),
        TRT_DESTROY_ENGINE(
                        new ExternalFunctionFactory("destroyEngine", "destroyEngine", "(sint32): sint32"),
                        true),
        TRT_DESTROY_EXECUTION_CONTEXT(
                        new ExternalFunctionFactory("destroyExecutionContext", "destroyExecutionContext", "(sint32): sint32"),
                        true);

        private final ExternalFunctionFactory factory;
        private final boolean checkError;

        public ExternalFunctionFactory getFunctionFactory() {
            return factory;
        }

        TensorRTFunctionNFI(ExternalFunctionFactory functionFactory, boolean checkError) {
            this.factory = functionFactory;
            this.checkError = checkError;
        }
    }

    class TRTFunction extends Function {

        private final ExternalFunctionFactory factory;
        private Function nfiFunction;

        TRTFunction(ExternalFunctionFactory factory) {
            super(factory.getName());
            this.factory = factory;
        }

        @Override
        public Object call(Object[] arguments) {
            try {
                if (nfiFunction == null) {
                    // load function symbol lazily
                    CompilerDirectives.transferToInterpreterAndInvalidate();
                    nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
                }
                Object result = INTEROP.execute(nfiFunction, arguments);
                checkTRTReturnCode(result, nfiFunction.getName());
                return result;
            } catch (InteropException e) {
                CompilerDirectives.transferToInterpreter();
                throw new RuntimeException(e);
            }
        }

    }

    class ErrorCheckedTRTFunction extends Function {
        private final ExternalFunctionFactory factory;
        private Function nfiFunction;

        ErrorCheckedTRTFunction(ExternalFunctionFactory factory) {
            super(factory.getName());
            this.factory = factory;
        }

        @Override
        @TruffleBoundary
        public Object call(Object[] arguments) {
            try {
                if (nfiFunction == null) {
                    // load function symbol lazily
                    CompilerDirectives.transferToInterpreterAndInvalidate();
                    nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
                }
                Object result = INTEROP.execute(nfiFunction, arguments);
                checkTRTReturnCode(result, nfiFunction.getName());
                return result;
            } catch (InteropException e) {
                CompilerDirectives.transferToInterpreter();
                throw new RuntimeException(e);
            }
        }
    }

    class EnqueueFunction extends Function {
        private final ExternalFunctionFactory factory;
        private Function nfiFunction;
        private TensorRTSetStreamFunction tensorRTSetStreamFunction;

        protected EnqueueFunction(ExternalFunctionFactory factory) {
            super(factory.getName());
            this.factory = factory;
        }

        public void setTensorRTSetStreamFunction (TensorRTSetStreamFunction tensorRTSetStreamFunction){
            this.tensorRTSetStreamFunction = tensorRTSetStreamFunction;
        }

        public Function registerTensorRTEnqueueFunction(){
            final Function wrapperFunction = new CUDALibraryFunction(factory.getName(), factory.getNFISignature()) {
                @Override
                @TruffleBoundary
                protected Object call(Object[] arguments) throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
                    checkArgumentLength(arguments, 3);
                    int engineHandle = expectInt(arguments[0]);
                    int batchSize = expectInt(arguments[1]);

                    // extract pointers from buffers array argument
                    Object bufferArg = arguments[2];
                    if (!INTEROP.hasArrayElements(bufferArg)) {
                        throw UnsupportedMessageException.create();
                    }
                    int numBuffers = (int) INTEROP.getArraySize(bufferArg);
                    try (UnsafeHelper.PointerArray pointerArray = UnsafeHelper.createPointerArray(numBuffers)) {
                        if (nfiFunction == null) {
                            // load function symbol lazily
                            CompilerDirectives.transferToInterpreterAndInvalidate();
                            nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
                        }
                        for (int i = 0; i < numBuffers; ++i) {
                            try {
                                Object buffer = INTEROP.readArrayElement(bufferArg, i);
                                if (!(buffer instanceof DeviceArray) && !(buffer instanceof GPUPointer)) {
                                    UnsupportedTypeException.create(new Object[]{buffer});
                                }
                                pointerArray.setValueAt(i, INTEROP.asPointer(buffer));
                            } catch (InvalidArrayIndexException e) {
                                InvalidArrayIndexException.create(i);
                            }
                        }
                        long stream = 0;
                        long eventConsumed = 0;
                        Object result = new CUDALibraryExecution(context.getGrCUDAExecutionContext(), nfiFunction, tensorRTSetStreamFunction, this.createComputationArgumentWithoutHandle(arguments)).schedule();
                        if (!INTEROP.fitsInInt(result)) {
                            CompilerDirectives.transferToInterpreter();
                            throw new RuntimeException("result of 'enqueue' is not an int");
                        }
                        return INTEROP.asInt(result) == 1;
                    }
                }
            };
            return wrapperFunction;
        }
    }
}
