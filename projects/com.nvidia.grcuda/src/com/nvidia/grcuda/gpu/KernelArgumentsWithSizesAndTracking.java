package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.ComputationArgument;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.array.DeviceArray;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.LibraryFactory;
import com.oracle.truffle.api.profiles.ValueProfile;

import java.io.Closeable;

public class KernelArgumentsWithSizesAndTracking extends KernelArgumentsWithSizes implements Closeable {

    public KernelArgumentsWithSizesAndTracking(Object[] args, ComputationArgument[] kernelArgumentList, DeviceArray sizesArray, DeviceArray trackingArray) {
        super(args, kernelArgumentList, args.length + 2, sizesArray);

        // Create a new argument that represents the OOB tracking array, and add it as last argument;
        UnsafeHelper.PointerObject pointer = UnsafeHelper.createPointerObject();
        pointer.setValueOfPointer(sizesArray.getPointer());
        this.setArgument(args.length + 1, pointer);
        // Store each array size in the new input array;
        for (int i = 0; i < trackingArray.getArraySize(); i++) {
            try {
                sizesArray.writeArrayElement(i, 0,
                        LibraryFactory.resolve(InteropLibrary.class).getUncached(),
                        ValueProfile.getUncached());
            } catch (InvalidArrayIndexException | UnsupportedTypeException e) {
                throw new GrCUDAInternalException("error initializing tracking array at index " + i);
            }
        }
    }
}
