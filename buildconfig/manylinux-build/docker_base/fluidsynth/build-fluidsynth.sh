#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

FSYNTH_VER="2.2.8"
FSYNTH="fluidsynth-$FSYNTH_VER"

curl -sL --retry 10 https://github.com/FluidSynth/fluidsynth/archive/v${FSYNTH_VER}.tar.gz > ${FSYNTH}.tar.gz
sha512sum -c fluidsynth.sha512
tar xzf ${FSYNTH}.tar.gz

cd $FSYNTH

# This hack is only needed for fluidsynth 2.2.x and can be removed once
# fluidsynth is updated and https://github.com/FluidSynth/fluidsynth/pull/978
# makes it to a release.
# Currently fluidsynth uses non-standard LIB_INSTALL_DIR instead of
# CMAKE_INSTALL_LIBDIR, but we set the latter.
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i 's/LIB_INSTALL_DIR/CMAKE_INSTALL_LIBDIR/g' CMakeLists.txt src/CMakeLists.txt
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # the -i flag on mac sed expects some kind of suffix (otherwise it errors)
    sed -i '' 's/LIB_INSTALL_DIR/CMAKE_INSTALL_LIBDIR/g' CMakeLists.txt src/CMakeLists.txt
fi

mkdir build
cd build

if [[ "$OSTYPE" == "darwin"* ]]; then
    # We don't need fluidsynth framework on mac builds
    export FLUIDSYNTH_EXTRA_MAC_FLAGS="-Denable-framework=NO"
fi

cmake .. $PG_BASE_CMAKE_FLAGS -Denable-readline=OFF $FLUIDSYNTH_EXTRA_MAC_FLAGS
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
