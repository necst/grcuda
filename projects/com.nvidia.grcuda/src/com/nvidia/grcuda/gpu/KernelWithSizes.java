package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.DeviceArray;
import com.nvidia.grcuda.ElementType;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.LibraryFactory;
import com.oracle.truffle.api.profiles.ValueProfile;

import java.util.ArrayList;
import java.util.List;

public final class KernelWithSizes extends Kernel {

    private final DeviceArray sizesArray;

    public KernelWithSizes(CUDARuntime cudaRuntime, String kernelName, CUDARuntime.CUModule kernelModule, long kernelFunction, String kernelSignature, int numOfPointers) {
        super(cudaRuntime, kernelName, kernelModule, kernelFunction, kernelSignature);
        this.sizesArray = new DeviceArray(cudaRuntime, numOfPointers, ElementType.LONG);
    }

    @Override
    KernelArguments createKernelArguments(Object[] args, InteropLibrary int32Access, InteropLibrary int64Access, InteropLibrary doubleAccess)
            throws UnsupportedTypeException, ArityException {
        // The number of provided arguments should be equal to the number of arguments in the modified signature + 1,
        // where the + 1 is the sizes array;
        if (args.length + 1 != this.getArgumentTypes().length) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(this.getArgumentTypes().length, args.length + 1);
        }
        KernelArguments kernelArgs = new KernelArguments(args.length + 1);

        // Add all the arguments provided by the user;
        for (int argIdx = 0; argIdx < args.length; argIdx++) {
            ArgumentType type = this.getArgumentTypes()[argIdx];
            processAndAddKernelArgument(type, kernelArgs, argIdx, args, int32Access, int64Access, doubleAccess);
        }
        // Add the additional arguments;
        List<Long> arraySizes = new ArrayList<>();
        for (Object arg : args) {
            if (arg instanceof DeviceArray) {
                DeviceArray deviceArray = (DeviceArray) arg;
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
                // This should be logged somehow;
                System.out.println("error setting size of array at index " + i + " , with value " + arraySizes.get(i));
            }
        }

        return kernelArgs;
    }
}