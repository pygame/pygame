#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

HARFBUZZ_VER=3.0.0
HARFBUZZ_NAME="harfbuzz-$HARFBUZZ_VER"
curl -sL --retry 10 https://github.com/harfbuzz/harfbuzz/releases/download/${HARFBUZZ_VER}/${HARFBUZZ_NAME}.tar.xz > ${HARFBUZZ_NAME}.tar.xz

sha512sum -c harfbuzz.sha512
unxz -xf ${HARFBUZZ_NAME}.tar.xz
tar -xf ${HARFBUZZ_NAME}.tar
cd ${HARFBUZZ_NAME}

# To avoid a circular dependency on freetype
./configure $ARCHS_CONFIG_FLAG --with-freetype=no --with-fontconfig=no
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
