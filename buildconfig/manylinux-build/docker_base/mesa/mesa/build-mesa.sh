#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

# We need mesa for opengl, gbm (SDL kmsdrm driver needs it), egl (SDL 
# wayland driver needs this) and glx (SDL needs it)
# we don't support vulkan yet

MESA_VER="mesa-22.2.0"
MESA="mesa-$MESA_VER" # yes mesa comes twice in the name

curl -sL --retry 10 https://gitlab.freedesktop.org/mesa/mesa/-/archive/$MESA_VER/$MESA.tar.gz > $MESA.tar.gz
sha512sum -c mesa.sha512sum

tar xzf $MESA.tar.gz
cd $MESA

# For now, we don't compile in LLVM because of its weight. Because of this, we
# can't compile in support for the radeonsi driver
if [ `uname -m` == "aarch64" ]; then
    # On aarch64 we allow mesa to use all drivers it wants to pick by default
    # (because radeonsi is not used on arm platforms)
    GALLIUM_DRIVERS="auto"  
else
    # all default except radeonsi
    GALLIUM_DRIVERS="r300,r600,nouveau,virgl,svga,swrast,iris,crocus,i915"
fi

# build with meson+ninja
meson build/ --buildtype=release -Dlibdir=lib \
    -Dgallium-drivers=$GALLIUM_DRIVERS -Dvulkan-drivers=[]
ninja -C build/ install
