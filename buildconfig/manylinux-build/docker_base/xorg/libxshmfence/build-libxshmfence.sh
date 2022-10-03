#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

XSHMFENCE_VER="libxshmfence-1.3"
XSHMFENCE="libxshmfence-$XSHMFENCE_VER" # yes libxshmfence is repeated

curl -sL --retry 10 https://gitlab.freedesktop.org/xorg/lib/libxshmfence/-/archive/$XSHMFENCE_VER/$XSHMFENCE.tar.gz > $XSHMFENCE.tar.gz
sha512sum -c libxshmfence.sha512sum

tar xzf $XSHMFENCE.tar.gz
cd $XSHMFENCE

./autogen.sh --disable-static
make
make install
