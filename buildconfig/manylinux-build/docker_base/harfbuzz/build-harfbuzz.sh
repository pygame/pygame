#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

HARFBUZZ_VER=3.0.0
HARFBUZZ_NAME="harfbuzz-$HARFBUZZ_VER"
curl -sL https://github.com/harfbuzz/harfbuzz/releases/download/${HARFBUZZ_VER}/${HARFBUZZ_NAME}.tar.xz > ${HARFBUZZ_NAME}.tar.xz

sha512sum -c harfbuzz.sha512
tar -xf ${HARFBUZZ_NAME}.tar.xz
cd ${HARFBUZZ_NAME}
./configure
make
make install
