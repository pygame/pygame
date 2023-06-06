#!/bin/bash
set -e -x

# https://wayland.freedesktop.org/building.html
# https://www.linuxfromscratch.org/blfs/view/svn/general/wayland.html
# https://www.linuxfromscratch.org/blfs/view/svn/general/wayland-protocols.html
# https://gitlab.freedesktop.org/libdecor/libdecor


cd $(dirname `readlink -f "$0"`)

WAYLAND_VER="1.21.0"
WAYLAND_PROTOCOLS_VER="1.31"
WAYLAND="wayland-${WAYLAND_VER}"
WAYLAND_PROTOCOLS="wayland-protocols-${WAYLAND_PROTOCOLS_VER}"
LIBDECOR_VER="0.1.1"
LIBDECOR="libdecor-${LIBDECOR_VER}"

curl -sL --retry 10 https://gitlab.freedesktop.org/wayland/wayland/-/releases/${WAYLAND_VER}/downloads/${WAYLAND}.tar.xz > ${WAYLAND}.tar.xz
curl -sL --retry 10 https://gitlab.freedesktop.org/wayland/wayland-protocols/-/releases/${WAYLAND_PROTOCOLS_VER}/downloads/${WAYLAND_PROTOCOLS}.tar.xz > ${WAYLAND_PROTOCOLS}.tar.xz
curl -sL --retry 10 https://gitlab.freedesktop.org/libdecor/libdecor/uploads/ee5ef0f2c3a4743e8501a855d61cb397/${LIBDECOR}.tar.xz > ${LIBDECOR}.tar.xz

sha512sum -c wayland.sha512

tar xf ${WAYLAND}.tar.xz
tar xf ${WAYLAND_PROTOCOLS}.tar.xz
tar xf ${LIBDECOR}.tar.xz


cd $WAYLAND
mkdir build
cd    build
meson setup ..            \
      --buildtype=release \
	  -Dtests=false \
	  -Dlibdir=lib \
      -Ddocumentation=false
ninja
ninja install

cd ../..


cd $WAYLAND_PROTOCOLS
mkdir build
cd build
meson setup .. \
		--buildtype=release \
		--prefix=/usr \
		-Dtests=false
ninja
ninja install

cd ../..

pkg-config --exists 'wayland-client >= 1.18'
pkg-config --exists wayland-scanner
pkg-config --exists wayland-egl
pkg-config --exists wayland-cursor
pkg-config --exists wayland-protocols
pkg-config --exists egl
pkg-config --exists 'xkbcommon >= 0.5.0'

cd $LIBDECOR

# Don't compile with cairo
sed -i 1d src/plugins/meson.build
meson build --buildtype release -Dinstall_demo=false -Ddemo=false --prefix=/usr
meson install -C build

cd ..
