#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

PORTMIDI="portmidi-src-217"
SRC_ZIP="${PORTMIDI}.zip"

curl -sL --retry 10 http://downloads.sourceforge.net/project/portmedia/portmidi/217/${SRC_ZIP} > ${SRC_ZIP}
sha512sum -c portmidi.sha512
unzip $SRC_ZIP

# if [ "$(uname -i)" = "x86_64" ]; then
#    export JAVA_HOME=/usr/lib/jvm/java-1.7.0-openjdk.x86_64
#    JRE_LIB_DIR=amd64
# else
#    export JAVA_HOME=/usr/lib/jvm/java-1.7.0-openjdk
#    JRE_LIB_DIR=i386
# fi
# ls ${JAVA_HOME}
# ls ${JAVA_HOME}/jre
# ls ${JAVA_HOME}/jre/lib
# ls ${JAVA_HOME}/jre/lib/$JRE_LIB_DIR
# ls ${JAVA_HOME}/jre/lib/$JRE_LIB_DIR/server

cd portmidi/
patch -p1 < ../no-java.patch
if [[ "$MAC_ARCH" == "arm64" ]]; then
    patch -p1 < ../mac_arm64.patch
elif [[ "$MAC_ARCH" == "x86_64" ]]; then
    patch -p1 < ../mac.patch
fi

# cmake -DJAVA_JVM_LIBRARY=${JAVA_HOME}/jre/lib/${JRE_LIB_DIR}/server/libjvm.so .
mkdir buildportmidi
cd buildportmidi

cmake -DCMAKE_BUILD_TYPE=Release .. $ARCHS_CONFIG_CMAKE_FLAG
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
