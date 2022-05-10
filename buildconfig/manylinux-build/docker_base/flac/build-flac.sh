#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

FLAC=flac-1.3.4

curl -sL --retry 10 http://downloads.xiph.org/releases/flac/${FLAC}.tar.xz > ${FLAC}.tar.xz
sha512sum -c flac.sha512

# The tar we have is too old to handle .tar.xz directly
unxz ${FLAC}.tar.xz
tar xf ${FLAC}.tar
cd $FLAC

./configure $ARCHS_CONFIG_FLAG
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
