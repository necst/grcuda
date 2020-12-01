package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.ComputationArgument;
import com.nvidia.grcuda.ComputationArgumentWithValue;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.array.DeviceArray;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.LibraryFactory;
import com.oracle.truffle.api.profiles.ValueProfile;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class KernelArgumentsWithSizes extends KernelArguments implements Closeable {

    public KernelArgumentsWithSizes(Object[] args, ComputationArgument[] kernelArgumentList, DeviceArray sizesArray) {
        this(args, kernelArgumentList, args.length + 1, sizesArray);
    }

    protected KernelArgumentsWithSizes(Object[] args, ComputationArgument[] kernelArgumentList, int extendedArgsLength, DeviceArray sizesArray) {
        super(args, kernelArgumentList, extendedArgsLength);

        // Add the additional arguments;
        List<Long> argsArraySizes = new ArrayList<>();
        for (Object arg : args) {
            if (arg instanceof AbstractArray) {
                AbstractArray deviceArray = (AbstractArray) arg;
                // Get the array size;
                argsArraySizes.add(deviceArray.getArraySize());
            }
        }

        // Create a new argument that represents the sizes array, and add it as last argument;
        UnsafeHelper.PointerObject pointer = UnsafeHelper.createPointerObject();
        pointer.setValueOfPointer(sizesArray.getPointer());
        this.setArgument(args.length, pointer);
        // Store each array size in the new input array;
        for (int i = 0; i < argsArraySizes.size(); i++) {
            try {
                sizesArray.writeArrayElement(i, argsArraySizes.get(i),
                        LibraryFactory.resolve(InteropLibrary.class).getUncached(),
                        ValueProfile.getUncached());
            } catch (InvalidArrayIndexException | UnsupportedTypeException e) {
                throw new GrCUDAInternalException("error setting size of array at index " + i + " , with value " + argsArraySizes.get(i));
            }
        }
    }
}
