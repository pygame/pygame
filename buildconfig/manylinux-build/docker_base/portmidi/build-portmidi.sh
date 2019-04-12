#!/bin/bash
set -e -x

cd /portmidi_build/

SRC_ZIP="portmidi-src-217.zip"

curl -sL http://downloads.sourceforge.net/project/portmedia/portmidi/217/${SRC_ZIP} > ${SRC_ZIP}
sha512sum -c portmidi.sha512
unzip $SRC_ZIP

cd portmidi/
patch -p1 < ../no-java.patch
#cmake -DJAVA_JVM_LIBRARY=${JAVA_HOME}/jre/lib/${JRE_LIB_DIR}/server/libjvm.so .
cmake -DCMAKE_BUILD_TYPE=Release .
make
make install
