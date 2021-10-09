#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

FREETYPE=freetype-2.11.0

if [ ! -d $FREETYPE ]; then

	curl -sL http://download.savannah.gnu.org/releases/freetype/${FREETYPE}.tar.gz > ${FREETYPE}.tar.gz
	sha512sum -c freetype.sha512

	tar xzf ${FREETYPE}.tar.gz
fi
cd $FREETYPE
./configure $EXTRA_CONFIG_FREETYPE
make
make install
