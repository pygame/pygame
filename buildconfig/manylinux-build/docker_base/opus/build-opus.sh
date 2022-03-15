#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

OPUS=opus-1.3.1

curl -sL --retry 10 https://archive.mozilla.org/pub/opus/${OPUS}.tar.gz > ${OPUS}.tar.gz
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
