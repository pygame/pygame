#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)
PULSEFILE="pulseaudio-14.0"

curl -sL --retry 10 https://www.freedesktop.org/software/pulseaudio/releases/${PULSEFILE}.tar.xz > ${PULSEFILE}.tar.xz
sha512sum -c pulseaudio.sha512
unxz ${PULSEFILE}.tar.xz
tar xf ${PULSEFILE}.tar

cd ${PULSEFILE}
./configure $ARCHS_CONFIG_FLAG --disable-manpages --disable-gsettings
make
make install
