package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.array.DeviceArray;
import com.nvidia.grcuda.gpu.computation.KernelWithSizesAndTrackingExecution;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public class ConfiguredKernelWithSizesAndTracking extends ConfiguredKernel {

    private final KernelWithSizesAndTracking kernel;

    public ConfiguredKernelWithSizesAndTracking(KernelWithSizesAndTracking kernel, KernelConfig config) {
        super(kernel, config);
        this.kernel = kernel;
    }

    @ExportMessage
    @Override
    @CompilerDirectives.TruffleBoundary
    Object execute(Object[] arguments,
                   @CachedLibrary(limit = "3") InteropLibrary boolAccess,
                   @CachedLibrary(limit = "3") InteropLibrary int8Access,
                   @CachedLibrary(limit = "3") InteropLibrary int16Access,
                   @CachedLibrary(limit = "3") InteropLibrary int32Access,
                   @CachedLibrary(limit = "3") InteropLibrary int64Access,
                   @CachedLibrary(limit = "3") InteropLibrary doubleAccess) throws UnsupportedTypeException, ArityException {
        kernel.incrementLaunchCount();
        try (KernelArgumentsWithSizesAndTracking args = kernel.createKernelArguments(arguments, boolAccess, int8Access, int16Access,
                int32Access, int64Access, doubleAccess)) {
            // If using a manually specified stream, do not schedule it automatically, but execute it immediately;
            if (!config.useCustomStream()) {
                new KernelWithSizesAndTrackingExecution(this, args).schedule();
            } else {
                kernel.getGrCUDAExecutionContext().getCudaRuntime().cuLaunchKernel(kernel, config, args, config.getStream());
            }
        }
        return this;
    }
}
