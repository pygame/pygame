#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

ZLIB_VER=1.2.12
ZLIB_NAME="zlib-$ZLIB_VER"
curl -sL --retry 10 https://www.zlib.net/${ZLIB_NAME}.tar.gz > ${ZLIB_NAME}.tar.gz

sha512sum -c zlib.sha512
tar -xf ${ZLIB_NAME}.tar.gz
cd ${ZLIB_NAME}

./configure $ARCHS_CONFIG_FLAG
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
