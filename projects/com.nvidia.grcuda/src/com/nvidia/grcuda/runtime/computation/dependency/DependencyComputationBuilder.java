package com.nvidia.grcuda.runtime.computation.dependency;

import com.nvidia.grcuda.ComputationArgumentWithValue;

import java.util.List;

public interface DependencyComputationBuilder {
    DependencyComputation initialize(List<ComputationArgumentWithValue> argumentList);
}
