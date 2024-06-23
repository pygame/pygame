#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)
GLIB=glib-2.80.0

curl -sL --retry 10 https://download.gnome.org/sources/glib/2.80/${GLIB}.tar.xz > ${GLIB}.tar.xz
sha512sum -c glib.sha512

unxz ${GLIB}.tar.xz
tar xf ${GLIB}.tar
cd $GLIB

if [[ "$MAC_ARCH" == "arm64" ]]; then
    # https://docs.gtk.org/glib/cross-compiling.html
    # https://mesonbuild.com/Cross-compilation.html
    # https://discourse.gnome.org/t/build-glib-fat-library-x86-64-and-arm64-on-macos-apple-m1/11092
    # https://clang.llvm.org/docs/CrossCompilation.html
    pwd
    export GLIB_COMPILE_MESON="--cross-file ../../../../macdependencies/macos-arm64.ini"
fi

if [[ "$OSTYPE" != "darwin"* ]]; then
    # glib needs a "python3"
    ln -s /opt/python/cp310-cp310/bin/python3.10 /usr/bin/python3
    python3 --version
fi

# configure the build, see https://gitlab.gnome.org/GNOME/glib/-/blob/main/docs/reference/glib/building.md
# also see for full list of options https://github.com/GNOME/glib/blob/main/meson_options.txt
meson setup $GLIB_COMPILE_MESON _build \
    -Dlibdir=lib \
    -Dbuildtype=release \
    -Ddefault_library=shared \
    -Ddocumentation=false \
    -Dman-pages=disabled \
    -Dselinux=disabled \
    -Ddtrace=false \
    -Dsystemtap=false \
    -Dnls=disabled \
    -Dtests=false \
    -Db_coverage=false

meson compile -C _build
meson install -C _build

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    # https://mesonbuild.com/Installing.html#destdir-support
    
    DESTDIR=$MACDEP_CACHE_PREFIX_PATH meson install --no-rebuild --only-changed -C _build && break || sleep 1
fi

if [[ "$OSTYPE" != "darwin"* ]]; then
    # clean up the python3 we made
    rm /usr/bin/python3
fi
