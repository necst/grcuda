#!/bin/sh
mx build;

# Install for Java 8+;
mkdir -p $GRAAL_HOME/languages/grcuda;
cp $GRCUDA_HOME/mxbuild/dists/grcuda.jar $GRAAL_HOME/languages/grcuda/.;
