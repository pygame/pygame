#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

HARFBUZZ_VER=3.0.0
HARFBUZZ_NAME="harfbuzz-$HARFBUZZ_VER"
curl -sL https://github.com/harfbuzz/harfbuzz/releases/download/${HARFBUZZ_VER}/${HARFBUZZ_NAME}.tar.xz > ${HARFBUZZ_NAME}.tar.xz

sha512sum -c harfbuzz.sha512
unxz -xf ${HARFBUZZ_NAME}.tar.xz
tar -xf ${HARFBUZZ_NAME}.tar
cd ${HARFBUZZ_NAME}
# To avoid a circular dependency on freetype
./configure --with-freetype=no --with-fontconfig=no
make
make install
