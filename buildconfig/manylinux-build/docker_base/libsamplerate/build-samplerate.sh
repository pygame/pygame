#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

SAMPLERATE_VERSION=0.2.2
SAMPLERATE=libsamplerate-${SAMPLERATE_VERSION}

curl -sL --retry 10 https://github.com/libsndfile/libsamplerate/releases/download/${SAMPLERATE_VERSION}/${SAMPLERATE}.tar.xz > ${SAMPLERATE}.tar.xz
sha512sum -c samplerate.sha512

tar xf ${SAMPLERATE}.tar.xz
cd $SAMPLERATE

./configure --prefix=/usr/local/ \
            --disable-static     \
            --docdir=/usr/local/share/doc/${SAMPLERATE} &&
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
