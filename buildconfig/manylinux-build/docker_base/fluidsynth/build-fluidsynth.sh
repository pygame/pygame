#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

FSYNTH_VERSION="v2.2.3"
FSYNTH="fluidsynth-2.2.3"

curl -sL https://github.com/FluidSynth/fluidsynth/archive/${FSYNTH_VERSION}.tar.gz > ${FSYNTH}.tar.gz
sha512sum -c fluidsynth.sha512
tar xzf ${FSYNTH}.tar.gz

cd $FSYNTH
mkdir build
cd build
cmake .. -Denable-readline=OFF
make
make install
