#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

GLIB=glib-2.56.4

curl -sL https://download.gnome.org/sources/glib/2.56/${GLIB}.tar.xz > ${GLIB}.tar.xz
sha512sum -c glib.sha512

unxz ${GLIB}.tar.xz
tar xzf ${GLIB}.tar
cd $GLIB
./configure --with-pcre=internal
make
make install
