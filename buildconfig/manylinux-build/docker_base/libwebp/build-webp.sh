#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

WEBP=libwebp-1.0.2

curl -sL http://storage.googleapis.com/downloads.webmproject.org/releases/webp/${WEBP}.tar.gz > ${WEBP}.tar.gz
sha512sum -c webp.sha512

tar xzf ${WEBP}.tar.gz
cd $WEBP
./configure
make
make install
