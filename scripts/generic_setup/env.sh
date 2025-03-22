#!/bin/bash

## make sure cuda is properly loaded/installed, e.g.:
# module load cuda/12.1

## (java workloads) maven installed/configured, e.g.:
# module load maven/3.8.4

## setup env
export INSTALL_DIR=$HOME/cf25_grcuda ## modify as necessary
mkdir -p $INSTALL_DIR

export LABSJDK_HOME=$INSTALL_DIR/labsjdk-ce-11.0.15-jvmci-22.1-b01
export JAVA_HOME=$LABSJDK_HOME
export GRCUDA_HOME=$INSTALL_DIR/grcuda
export GRAAL_HOME=$INSTALL_DIR/graalvm-ce-java11-22.1.0

export PATH=$INSTALL_DIR/mx:$PATH
export PATH=$GRAAL_HOME/bin:$PATH
export PATH=$JAVA_HOME/bin:$PATH
