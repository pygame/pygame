#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

MODPLUG_VER=0.8.9.0
MODPLUG_NAME="libmodplug-$MODPLUG_VER"
curl -sL --retry 10 https://downloads.sourceforge.net/modplug-xmms/${MODPLUG_NAME}.tar.gz > ${MODPLUG_NAME}.tar.gz


sha512sum -c libmodplug.sha512
tar -xf ${MODPLUG_NAME}.tar.gz
cd ${MODPLUG_NAME}

patch -Np1 -i ../libmodplug-0.8.9.0-no-fast-math.patch
autoreconf -vfi

./configure $ARCHS_CONFIG_FLAG
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
