#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

ALSA=alsa-lib-1.1.8

curl -sL ftp://ftp.alsa-project.org/pub/lib/${ALSA}.tar.bz2 > ${ALSA}.tar.bz2
sha512sum -c alsa.sha512

tar xjf ${ALSA}.tar.bz2
cd ${ALSA}
./configure --with-configdir=/usr/share/alsa
make
make install
