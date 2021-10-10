#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

ZLIB_VER=1.2.11
ZLIB_NAME="zlib-$ZLIB_VER"
curl -sL https://www.zlib.net/${ZLIB_NAME}.tar.gz > ${ZLIB_NAME}.tar.gz

sha512sum -c zlib.sha512
tar -xf ${ZLIB_NAME}.tar.gz
cd ${ZLIB_NAME}
./configure
make
make install
