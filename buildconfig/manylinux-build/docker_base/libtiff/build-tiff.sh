#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

TIFF=tiff-4.3.0

curl -sL --retry 10 https://download.osgeo.org/libtiff/${TIFF}.tar.gz > ${TIFF}.tar.gz
sha512sum -c tiff.sha512

tar xzf ${TIFF}.tar.gz
cd $TIFF

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    ./configure --disable-lzma --disable-webp --disable-zstd
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Use CMake on MacOS because arm64 builds fail with weird errors in ./configure
    cmake . $ARCHS_CONFIG_CMAKE_FLAG -DCMAKE_BUILD_TYPE=Release -Dlzma=OFF -Dwebp=OFF -Dzstd=OFF
fi

make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
