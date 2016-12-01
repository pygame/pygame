#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

FLAC=flac-1.3.1

curl -sL http://downloads.xiph.org/releases/flac/${FLAC}.tar.xz > ${FLAC}.tar.xz
sha512sum -c flac.sha512

# The tar we have is too old to handle .tar.xz directly
unxz ${FLAC}.tar.xz
tar xf ${FLAC}.tar
cd $FLAC
./configure
make
make install
