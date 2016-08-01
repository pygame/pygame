#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

FSYNTH="fluidsynth-1.1.6"

curl -sL https://sourceforge.net/projects/fluidsynth/files/${FSYNTH}/${FSYNTH}.tar.gz/download > ${FSYNTH}.tar.gz
sha512sum -c fluidsynth.sha512
tar xzf ${FSYNTH}.tar.gz

cd $FSYNTH
mkdir build
cd build
cmake ..
make
make install
