#!/bin/bash
set -e -x

# wayland requirement
# https://www.linuxfromscratch.org/blfs/view/svn/general/libxml2.html

cd $(dirname `readlink -f "$0"`)

LIBXML2_VER="2.10.3"
LIBXML2="libxml2-${LIBXML2_VER}"
curl -sL --retry 10 https://download.gnome.org/sources/libxml2/2.10/${LIBXML2}.tar.xz > ${LIBXML2}.xz

sha512sum -c libxml2.sha512

tar xf ${LIBXML2}.xz
cd $LIBXML2
./configure \
            --sysconfdir=/etc       \
            --disable-static        \
            --with-history          \
            --with-python=no		\
			# --prefix=/usr           \
            # PYTHON=/usr/bin/python3 \
            # --docdir=/usr/share/doc/libxml2-2.10.3 &&
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi

cd ..
