#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

SNDFILEVER=1.0.31
SNDFILE="libsndfile-$SNDFILEVER.tar.bz2"
curl -sL https://github.com/libsndfile/libsndfile/releases/download/${SNDFILEVER}/${SNDFILE} > ${SNDFILE}

sha512sum -c sndfile.sha512
tar xf ${SNDFILE}
cd libsndfile-${SNDFILEVER}
# autoreconf -fvi
./configure
make
make install
