#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

GLSLANG_VER=11.11.0
GLSLANG="glslang-$GLSLANG_VER"
curl -sL --retry 10 https://github.com/KhronosGroup/glslang/archive/refs/tags/${GLSLANG_VER}.tar.gz > ${GLSLANG}.tar.gz

sha512sum -c glslang.sha512sum
tar xzf ${GLSLANG}.tar.gz
cd $GLSLANG

mkdir build
cd build

cmake .. $PG_BASE_CMAKE_FLAGS
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
