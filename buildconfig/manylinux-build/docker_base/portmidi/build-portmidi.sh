#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

PORTMIDI_VER="2.0.3"
PORTMIDI="portmidi-${PORTMIDI_VER}"

curl -sL --retry 10 https://github.com/PortMidi/portmidi/archive/refs/tags/v${PORTMIDI_VER}.tar.gz> ${PORTMIDI}.tar.gz
sha512sum -c portmidi.sha512

tar xzf ${PORTMIDI}.tar.gz
cd $PORTMIDI

cmake . $PG_BASE_CMAKE_FLAGS
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
