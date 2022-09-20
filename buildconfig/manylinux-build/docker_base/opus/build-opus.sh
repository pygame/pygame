#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

OPUS=opus-1.3.1
OPUS_FILE=opusfile-0.12

curl -sL --retry 10 http://downloads.xiph.org/releases/opus/${OPUS}.tar.gz > ${OPUS}.tar.gz
curl -sL --retry 10 http://downloads.xiph.org/releases/opus/${OPUS_FILE}.tar.gz > ${OPUS_FILE}.tar.gz
sha512sum -c opus.sha512

tar xzf ${OPUS}.tar.gz
cd $OPUS

./configure $ARCHS_CONFIG_FLAG
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi

cd ..

tar xzf ${OPUS_FILE}.tar.gz
cd $OPUS_FILE

./configure $ARCHS_CONFIG_FLAG --disable-http
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi