#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

MPG123="mpg123-1.30.2"

curl -sL --retry 10 https://downloads.sourceforge.net/sourceforge/mpg123/${MPG123}.tar.bz2 > ${MPG123}.tar.bz2
sha512sum -c mpg123.sha512

bzip2 -d ${MPG123}.tar.bz2
tar xf ${MPG123}.tar
cd $MPG123

./configure $ARCHS_CONFIG_FLAG --enable-int-quality --disable-debug
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi

cd ..
