#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

WEBP=libwebp-1.3.2

curl -sL --retry 10 http://storage.googleapis.com/downloads.webmproject.org/releases/webp/${WEBP}.tar.gz > ${WEBP}.tar.gz
sha512sum -c webp.sha512

tar xzf ${WEBP}.tar.gz
cd $WEBP

./configure $ARCHS_CONFIG_FLAG
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
