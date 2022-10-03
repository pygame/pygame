#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

MACROS_VER="util-macros-1.19.3"
MACROS="macros-$MACROS_VER"

curl -sL --retry 10 https://gitlab.freedesktop.org/xorg/util/macros/-/archive/$MACROS_VER/$MACROS.tar.gz > $MACROS.tar.gz
sha512sum -c macros.sha512sum

tar xzf $MACROS.tar.gz
cd $MACROS

./autogen.sh
make
make install
