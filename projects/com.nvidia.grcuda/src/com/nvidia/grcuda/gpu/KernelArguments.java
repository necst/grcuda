package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.ComputationArgument;
import com.nvidia.grcuda.ComputationArgumentWithValue;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class KernelArguments implements Closeable {

    protected final Object[] originalArgs;
    /**
     * Associate each input object to the characteristics of its argument, such as its type and if it's constant;
     */
    protected final List<ComputationArgumentWithValue> kernelArgumentWithValues = new ArrayList<>();
    protected final UnsafeHelper.PointerArray argumentArray;
    protected final ArrayList<Closeable> argumentValues = new ArrayList<>();

    public KernelArguments(Object[] args, ComputationArgument[] kernelArgumentList) {
        this(args, kernelArgumentList, args.length);
    }

    protected KernelArguments(Object[] args, ComputationArgument[] kernelArgumentList, int argSize) {
        this.originalArgs = args;
        this.argumentArray = UnsafeHelper.createPointerArray(argSize);
        assert(args.length == kernelArgumentList.length);
        assert(argSize >= args.length);
        // Initialize the list of arguments and object references;
        for (int i = 0; i < args.length; i++) {
            kernelArgumentWithValues.add(new ComputationArgumentWithValue(kernelArgumentList[i], args[i]));
        }
    }

    public void setArgument(int argIdx, UnsafeHelper.MemoryObject obj) {
        argumentArray.setValueAt(argIdx, obj.getAddress());
        argumentValues.add(obj);
    }

    long getPointer() {
        return argumentArray.getAddress();
    }

    public Object[] getOriginalArgs() {
        return originalArgs;
    }

    public Object getOriginalArg(int index) {
        return originalArgs[index];
    }

    public List<ComputationArgumentWithValue> getKernelArgumentWithValues() {
        return kernelArgumentWithValues;
    }

    @Override
    public String toString() {
        return "KernelArgs=" + Arrays.toString(originalArgs);
    }

    @Override
    public void close() {
        this.argumentArray.close();
        for (Closeable c : argumentValues) {
            try {
                c.close();
            } catch (IOException e) {
                /* ignored */
            }
        }
    }
}
