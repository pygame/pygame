#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

GLIB=glib-2.70.0

curl -sL https://download.gnome.org/sources/glib/2.70/${GLIB}.tar.xz > ${GLIB}.tar.xz
sha512sum -c glib.sha512

tar xzf ${GLIB}.tar.xz
cd $GLIB
meson _build -v
ninja -C _build -v
ninja -C _build install -v
