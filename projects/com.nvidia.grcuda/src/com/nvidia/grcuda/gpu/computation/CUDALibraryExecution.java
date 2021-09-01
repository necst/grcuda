package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.ComputationArgumentWithValue;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.gpu.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.DefaultStream;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.ArrayList;
import java.util.List;

import static com.nvidia.grcuda.functions.Function.INTEROP;

/**
 * Computational element that wraps calls to CUDA libraries such as cuBLAS or cuML.
 */
public class CUDALibraryExecution extends GrCUDAComputationalElement {

    private final Function nfiFunction;
    private final Object[] args;

    public CUDALibraryExecution(AbstractGrCUDAExecutionContext context, Function nfiFunction, Object[] args) {
        super(context, new CUDALibraryExecutionInitializer());
        this.nfiFunction = nfiFunction;
        this.args = args;
    }

    @Override
    public Object execute() throws UnsupportedTypeException {
        // FIXME: don't sync before execution!
        this.grCUDAExecutionContext.getCudaRuntime().cudaDeviceSynchronize();

        // Execution happens on the default stream;
        Object result = null;
        try {
            result = INTEROP.execute(this.nfiFunction, this.args);
        } catch (ArityException | UnsupportedMessageException e) {
            System.out.println("error in execution of cuBLAS function");
            e.printStackTrace();
        }
        // Synchronize only the default stream;
        // FIXME: don't sync entire device, just the default stream!
        this.grCUDAExecutionContext.getCudaRuntime().cudaDeviceSynchronize();
//        this.grCUDAExecutionContext.getCudaRuntime().cudaStreamSynchronize(DefaultStream.get());
        this.setComputationFinished();
        return result;
    }

    private static class CUDALibraryExecutionInitializer implements InitializeArgumentList {
        CUDALibraryExecutionInitializer() {
        }

        @Override
        public List<ComputationArgumentWithValue> initialize() {
            return new ArrayList<>();
        }
    }
}
