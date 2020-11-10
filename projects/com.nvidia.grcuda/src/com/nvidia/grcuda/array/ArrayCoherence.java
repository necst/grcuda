package com.nvidia.grcuda.array;
/**
 * Using memory Coherence protocol for arrays similar to the one on distributed computing systems:
 * - Shared: read only
 * - Invalid: invalidated
 * - Exclusive: for read/write
 * */
public enum ArrayCoherence {
    SHARED,
    INVALID,
    EXCLUSIVE,
    MODIFIED
}
