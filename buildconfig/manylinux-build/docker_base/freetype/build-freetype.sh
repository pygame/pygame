#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

FREETYPE=freetype-2.11.0

if [ ! -d $FREETYPE ]; then

	curl -sL --retry 10 http://download.savannah.gnu.org/releases/freetype/${FREETYPE}.tar.gz > ${FREETYPE}.tar.gz
	sha512sum -c freetype.sha512

	tar xzf ${FREETYPE}.tar.gz
fi
cd $FREETYPE

./configure $EXTRA_CONFIG_FREETYPE $ARCHS_CONFIG_FLAG
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
