#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

PCIACCESS_VER="libpciaccess-0.16"
PCIACCESS="libpciaccess-$PCIACCESS_VER" # yes libpciaccess is repeated

curl -sL --retry 10 https://gitlab.freedesktop.org/xorg/lib/libpciaccess/-/archive/$PCIACCESS_VER/$PCIACCESS.tar.gz > $PCIACCESS.tar.gz
sha512sum -c libpciaccess.sha512sum

tar xzf $PCIACCESS.tar.gz
cd $PCIACCESS

./autogen.sh --disable-static
make
make install
