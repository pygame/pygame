#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

ALSA=alsa-lib-1.2.7.2
curl -sL https://www.alsa-project.org/files/pub/lib/${ALSA}.tar.bz2 > ${ALSA}.tar.bz2
sha512sum -c alsa.sha512

tar xjf ${ALSA}.tar.bz2
cd ${ALSA}

# alsa prefers /usr prefix as a default, so we explicitly override it
./configure --prefix=/usr/local --with-configdir=/usr/local/share/alsa 
make
make install
