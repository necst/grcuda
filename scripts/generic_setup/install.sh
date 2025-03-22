#!/bin/sh
source env.sh

# build needs to be perfomed with labsjdk
export JAVA_HOME=$LABSJDK_HOME
mx build;

# Install (moving the jar to the languages folder of graalvm)
mkdir -p $GRAAL_HOME/languages/grcuda;
cp $GRCUDA_HOME/mxbuild/dists/grcuda.jar $GRAAL_HOME/languages/grcuda/.;


# (java workloads) enable running them in offline-mode (e.g, for compute nodes of a cluster)
export JAVA_HOME=$GRAAL_HOME
cd $GRCUDA_HOME/projects/resources/java/grcuda-benchmark
mvn clean
mvn dependency:get -Dartifact=org.apache.maven.surefire:surefire-junit4:2.22.1 -Dtransitive=true
mvn dependency:resolve-plugins
mvn dependency:go-offline
mvn test-compile
mvn surefire:test -DskipTests