package com.nvidia.grcuda.gpu;

import java.util.Hashtable;

public abstract class ProfilableElement {
    private final boolean profilable;
    // contains latest execution time associated to the GPU on which it was executed
    Hashtable<Integer, Float> collectionOfExecution;
    public ProfilableElement(boolean profilable){
        collectionOfExecution = new Hashtable<Integer, Float>();
        this.profilable = profilable;
    }

    public boolean isProfilable(){
        return this.profilable;
    }

    public void addExecutionTime(int deviceId, float executionTime ){
        collectionOfExecution.put(deviceId, executionTime);
    }

    public float getExecutionTimeOnDevice(int deviceId){
        if(collectionOfExecution.get(deviceId) == null){
            return (float) 0.0;
        }else{
            return collectionOfExecution.get(deviceId);
        }
        
    }



    

}
