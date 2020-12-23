#!/bin/bash
set -e -x

cd /pulseaudio_build/
PULSEFILE="pulseaudio-14.0"

curl -sL https://www.freedesktop.org/software/pulseaudio/releases/${PULSEFILE}.tar.xz > ${PULSEFILE}.tar.xz
sha512sum -c pulseaudio.sha512
unxz ${PULSEFILE}.tar.xz
tar xf ${PULSEFILE}.tar

cd ${PULSEFILE}
./configure --disable-manpages --disable-gsettings
make
make install
