#!/bin/bash
set -e -x

cd /sndfile_build/
SNDFILEVER=1.0.30
SNDFILE="libsndfile-$SNDFILEVER.tar.bz2"
curl -sL https://github.com/libsndfile/libsndfile/releases/download/v${SNDFILEVER}/${SNDFILE} > ${SNDFILE}
# https://github.com/libsndfile/libsndfile/releases/download/v1.0.30/libsndfile-1.0.30.tar.bz2

sha512sum -c sndfile.sha512
tar xf ${SNDFILE}
cd libsndfile-${SNDFILEVER}
./configure
make
make install
