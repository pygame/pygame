#!/bin/bash
set -e -x

cd /portmidi_build/

SRC_ZIP="portmidi-src-217.zip"

curl -sL http://downloads.sourceforge.net/project/portmedia/portmidi/217/${SRC_ZIP} > ${SRC_ZIP}
sha512sum -c portmidi.sha512
unzip $SRC_ZIP

if [ "$(uname -i)" = "x86_64" ]; then
    export JAVA_HOME=/usr/lib/jvm/java-1.7.0-openjdk.x86_64
    JRE_LIB_DIR=amd64
else
    export JAVA_HOME=/usr/lib/jvm/java-1.7.0-openjdk
    JRE_LIB_DIR=i386
fi
ls ${JAVA_HOME}
ls ${JAVA_HOME}/jre
ls ${JAVA_HOME}/jre/lib
ls ${JAVA_HOME}/jre/lib/$JRE_LIB_DIR
ls ${JAVA_HOME}/jre/lib/$JRE_LIB_DIR/server

cd portmidi/
#cmake -DJAVA_JVM_LIBRARY=${JAVA_HOME}/jre/lib/${JRE_LIB_DIR}/server/libjvm.so .
cmake .
make
make install
