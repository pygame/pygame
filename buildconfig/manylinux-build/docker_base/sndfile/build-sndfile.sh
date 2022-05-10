#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

SNDFILEVER=1.1.0
SNDNAME="libsndfile-$SNDFILEVER"
SNDFILE="$SNDNAME.tar.xz"
curl -sL --retry 10 https://github.com/libsndfile/libsndfile/releases/download/${SNDFILEVER}/${SNDFILE} > ${SNDFILE}

sha512sum -c sndfile.sha512
tar xf ${SNDFILE}
cd $SNDNAME
# autoreconf -fvi

./configure $ARCHS_CONFIG_FLAG
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
