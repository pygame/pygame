#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

LIBXML2_VER=2.10.2
LIBXML2="libxml2-$LIBXML2_VER"
LIBXML2_DIR_SUFFIX=21b24b51608d471bb9f7c4225e23d0db2acecc52

curl -sL --retry 10 https://gitlab.gnome.org/GNOME/libxml2/-/archive/v${LIBXML2_VER}/libxml2-$LIBXML2_VER.tar.gz > ${LIBXML2}.tar.gz
sha512sum -c libxml2.sha512

tar xzf ${LIBXML2}.tar.gz
cd libxml2-v$LIBXML2_VER-$LIBXML2_DIR_SUFFIX

./autogen.sh $ARCHS_CONFIG_FLAG --disable-static --with-python=no
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
