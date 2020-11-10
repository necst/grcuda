package com.nvidia.grcuda.array;

public class CoherenceState {
    private ArrayCoherence state;


    public CoherenceState(){
        state = ArrayCoherence.EXCLUSIVE;
    }

    public void arrayIsWrote(){
        state = ArrayCoherence.MODIFIED;
    }

    public void arrayIsShared(){
        state = ArrayCoherence.SHARED;
    }

    public void arrayIsExclusive(){
        state = ArrayCoherence.EXCLUSIVE;
    }

    public void arrayIsInvalid(){
        state = ArrayCoherence.INVALID;
    }

    public ArrayCoherence getState(){
        return this.state;
    }


}
