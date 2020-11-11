package com.nvidia.grcuda.gpu.stream;

public class DefaultStream extends CUDAStream {
    
    static final int DEFAULT_STREAM_NUMBER = -1;
    static final int DEFAULT_DEVICE = 0;
    private static final DefaultStream defaultStream = new DefaultStream();
    
    private DefaultStream() {
        super(0, DEFAULT_STREAM_NUMBER,DEFAULT_DEVICE);
    }

    public static DefaultStream get() { return defaultStream; }

    @Override
    public boolean isDefaultStream() {
        return true; }

    @Override
    public String toString() {
        return "DefaultCUDAStream(streamNumber=" + DEFAULT_STREAM_NUMBER + "; address=0x" + Long.toHexString(this.getRawPointer()) + ")";
    }
}
