#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

FREETYPE=freetype-2.9.1

curl -sL http://download.savannah.gnu.org/releases/freetype/${FREETYPE}.tar.gz > ${FREETYPE}.tar.gz
sha512sum -c freetype.sha512

tar xzf ${FREETYPE}.tar.gz
cd $FREETYPE
./configure
make
make install
