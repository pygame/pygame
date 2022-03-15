#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

BZIP2_VER=1.0.8
BZIP2=bzip2-$BZIP2_VER

curl -sL --retry 10 https://sourceware.org/pub/bzip2/${BZIP2}.tar.gz > ${BZIP2}.tar.gz
sha512sum -c bzip2.sha512

tar xzf ${BZIP2}.tar.gz
cd $BZIP2

if [[ -z "${CC}" ]]; then
    make install
else
    # pass CC explicitly because it's needed here
    make install CC="${CC}"
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install PREFIX=${MACDEP_CACHE_PREFIX_PATH}/usr/local
fi

if [[ "$MAC_ARCH" == "arm64" ]]; then
    # We don't need bzip2 arm64 binaries, remove them so that intel binaries
    # are used correctly
    sudo rm /usr/local/bin/bzip2
    rm ${MACDEP_CACHE_PREFIX_PATH}/usr/local/bin/bzip2
fi
