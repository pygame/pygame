#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

# The latest libdecor release does not build on manylinux (due to it requiring
# very new libc API)
# So we instead use a more recent libdecor (but pin on the commit for build
# stability)
LIBDECOR_VER=8b42120d2b144bb9b5b62212d74d93ee23c82395
LIBDECOR="libdecor-$LIBDECOR_VER"

curl -sL --retry 10 https://gitlab.gnome.org/jadahl/libdecor/-/archive/$LIBDECOR_VER/$LIBDECOR.tar.gz > $LIBDECOR.tar.gz
sha512sum -c libdecor.sha512sum

tar xzf $LIBDECOR.tar.gz
cd $LIBDECOR

# This is a hack because I'm lazy :)
# libdecor depends on cairo and pango which are both kinda heavy dependencies
# and need more scripting work. But libdecor shared lib is not actually present
# in the manylinux wheel and is dynamically loaded by SDL at runtime (if
# available on users system)
# So we override the plugin builds to skip cairo (and only build dummy plugin)
echo "subdir('dummy')" > src/plugins/meson.build

meson build/ --buildtype=release -Dlibdir=lib -Ddemo=false
ninja -C build/ install
