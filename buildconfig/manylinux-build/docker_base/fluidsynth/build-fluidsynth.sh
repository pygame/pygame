#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

FSYNTH_VERSION="v2.2.3"
FSYNTH="fluidsynth-2.2.3"

curl -sL --retry 10 https://github.com/FluidSynth/fluidsynth/archive/${FSYNTH_VERSION}.tar.gz > ${FSYNTH}.tar.gz
sha512sum -c fluidsynth.sha512
tar xzf ${FSYNTH}.tar.gz

cd $FSYNTH
mkdir build
cd build

if [[ "$OSTYPE" == "darwin"* ]]; then
    # We don't need fluidsynth framework on mac builds
    export FLUIDSYNTH_EXTRA_MAC_FLAGS="-Denable-framework=NO"
fi

cmake .. -Denable-readline=OFF -DCMAKE_BUILD_TYPE=Release \
    $ARCHS_CONFIG_CMAKE_FLAG $FLUIDSYNTH_EXTRA_MAC_FLAGS
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
