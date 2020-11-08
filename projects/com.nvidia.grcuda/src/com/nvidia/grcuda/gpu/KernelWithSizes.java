package com.nvidia.grcuda.gpu;


import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.array.DeviceArray;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.gpu.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.LibraryFactory;
import com.oracle.truffle.api.profiles.ValueProfile;

import java.util.ArrayList;
import java.util.List;

public final class KernelWithSizes extends Kernel {

    private final DeviceArray sizesArray;

    public KernelWithSizes(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String kernelName, String kernelSymbol,
                           long kernelFunction, String kernelSignature, CUDARuntime.CUModule module, int numOfPointers) {
        super(grCUDAExecutionContext, kernelName, kernelSymbol, kernelFunction, kernelSignature, module);
        this.sizesArray = new DeviceArray(grCUDAExecutionContext, numOfPointers, Type.UINT64);
    }

    public KernelWithSizes(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String kernelName, String kernelSymbol,
                           long kernelFunction, String kernelSignature, CUDARuntime.CUModule module, String ptx, int numOfPointers) {
        super(grCUDAExecutionContext, kernelName, kernelSymbol, kernelFunction, kernelSignature, module, ptx);
        this.sizesArray = new DeviceArray(grCUDAExecutionContext, numOfPointers, Type.UINT64);
    }

    @Override
    protected KernelArguments createKernelArguments(Object[] args, InteropLibrary booleanAccess,
                                          InteropLibrary int8Access, InteropLibrary int16Access,
                                          InteropLibrary int32Access, InteropLibrary int64Access, InteropLibrary doubleAccess)
            throws UnsupportedTypeException, ArityException {
        // The number of provided arguments should be equal to the number of arguments in the modified signature - 1,
        // as we added the sizes array to the modified signature;
        if (args.length + 1 != this.kernelComputationArguments.length) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(this.kernelComputationArguments.length, args.length + 1);
        }
        KernelArguments kernelArgs = new KernelArguments(args, this.kernelComputationArguments);

        // Add all the arguments provided by the user;
        for (int argIdx = 0; argIdx < args.length; argIdx++) {
            processAndAddKernelArgument(kernelArgs, args, argIdx, booleanAccess, int8Access, int16Access, int32Access, int64Access, doubleAccess);
        }
        // Add the additional arguments;
        List<Long> arraySizes = new ArrayList<>();
        for (Object arg : args) {
            if (arg instanceof AbstractArray) {
                AbstractArray deviceArray = (AbstractArray) arg;
                // Get the array size;
                arraySizes.add(deviceArray.getArraySize());
            }
        }

        // Create a new argument that represents the sizes array, and add it as last argument;
        UnsafeHelper.PointerObject pointer = UnsafeHelper.createPointerObject();
        pointer.setValueOfPointer(sizesArray.getPointer());
        kernelArgs.setArgument(args.length, pointer);
        // Store each array size in the new input array;
        for (int i = 0; i < arraySizes.size(); i++) {
            try {
                sizesArray.writeArrayElement(i, arraySizes.get(i),
                        LibraryFactory.resolve(InteropLibrary.class).getUncached(),
                        ValueProfile.getUncached());
            } catch (InvalidArrayIndexException e) {
                throw new GrCUDAInternalException("error setting size of array at index " + i + " , with value " + arraySizes.get(i));
            }
        }
        return kernelArgs;
    }
}
