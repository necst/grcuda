/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.nvidia.grcuda.runtime;

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