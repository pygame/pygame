#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

JPEG_VERSION=2.1.4
JPEG="libjpeg-turbo-${JPEG_VERSION}"

curl -sL --retry 10 https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/${JPEG_VERSION}.tar.gz > ${JPEG}.tar.gz

sha512sum -c libjpegturbo.sha512
tar xzf ${JPEG}.tar.gz
cd ${JPEG}

cmake . $PG_BASE_CMAKE_FLAGS -DWITH_TURBOJPEG=0

make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
