#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

PNG=libpng-1.6.37

curl -sL --retry 10 http://download.sourceforge.net/libpng/${PNG}.tar.gz > ${PNG}.tar.gz
sha512sum -c png.sha512

tar xzf ${PNG}.tar.gz
cd $PNG

./configure --with-zlib-prefix=/usr/local/ $ARCHS_CONFIG_FLAG
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
