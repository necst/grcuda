#!/bin/bash
source env.sh

### GrCUDA
cd $INSTALL_DIR
if [ ! -d "grcuda" ]; then
    echo "Cloning GrCUDA..."
    git clone https://github.com/necst/grcuda.git
else
    echo "GrCUDA already present, skipping clone."
fi

### GRAALVM
cd $INSTALL_DIR
wget https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-22.1.0/graalvm-ce-java11-linux-amd64-22.1.0.tar.gz
tar xfz graalvm-ce-java11-linux-amd64-22.1.0.tar.gz
rm graalvm-ce-java11-linux-amd64-22.1.0.tar.gz

### LABSJDK (to build from source)
cd $INSTALL_DIR
wget https://github.com/graalvm/labs-openjdk-11/releases/download/jvmci-22.1-b01/labsjdk-ce-11.0.15+2-jvmci-22.1-b01-linux-amd64.tar.gz
tar xfz labsjdk-ce-11.0.15+2-jvmci-22.1-b01-linux-amd64.tar.gz
rm labsjdk-ce-11.0.15+2-jvmci-22.1-b01-linux-amd64.tar.gz

### MX
cd $INSTALL_DIR
git clone https://github.com/graalvm/mx.git
cd $INSTALL_DIR/mx
git checkout 722b86b8ef87fbb297f7e33ee6014bbbd3f4a3a8

### GRAAL (to develop)
cd $INSTALL_DIR
git clone https://github.com/oracle/graal.git
cd $INSTALL_DIR/graal
git checkout 84541b16ae8a8726a0e7d76c7179d94a57ed84ee

## install the necessary components of graalvm
gu available
gu install native-image
gu install llvm-toolchain
gu install python 
gu rebuild-images polyglot

### (python benchmarks) create a graalpython env
graalpython -m venv $INSTALL_DIR/graalpython_venv
source $INSTALL_DIR/graalpython_venv/bin/activate
graalpython -m ginstall install setuptools;
graalpython -m ginstall install Cython;
graalpython -m ensurepip;
pip install numpy==1.16.4