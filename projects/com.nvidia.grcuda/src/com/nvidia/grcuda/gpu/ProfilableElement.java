package com.nvidia.grcuda.gpu;

public abstract class ProfilableElement {
    private final boolean profilable;

    public ProfilableElement(boolean profilable){
        this.profilable = profilable;
    }

    public boolean isProfilable(){
        return this.profilable;
    }


    

}
