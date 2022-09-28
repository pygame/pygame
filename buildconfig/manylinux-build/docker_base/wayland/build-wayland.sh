#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

WAYLAND_VER=1.21.0
WAYLAND_PROT_VER=1.26

WAYLAND="wayland-$WAYLAND_VER"
WAYLAND_PROT="wayland-protocols-$WAYLAND_PROT_VER"

curl -sL --retry 10 https://gitlab.freedesktop.org/wayland/wayland/-/archive/$WAYLAND_VER/$WAYLAND.tar.gz > $WAYLAND.tar.gz
curl -sL --retry 10 https://gitlab.freedesktop.org/wayland/wayland-protocols/-/archive/$WAYLAND_PROT_VER/$WAYLAND_PROT.tar.gz > $WAYLAND_PROT.tar.gz

sha512sum -c wayland.sha512sum

tar xzf $WAYLAND.tar.gz
cd $WAYLAND

meson build/ --buildtype=release -Dlibdir=lib -Ddocumentation=false -Dtests=false
ninja -C build/ install

cd ..
tar xzf $WAYLAND_PROT.tar.gz
cd $WAYLAND_PROT

meson build/ --buildtype=release -Dtests=false
ninja -C build/ install
