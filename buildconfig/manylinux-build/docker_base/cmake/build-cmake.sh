#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

# The latest cmake doesn't compile on centos 5.
# This version does, and works with latest fluidsynth(2.2.3).
CMAKE=cmake-3.12.4

if [ ! -d $CMAKE ]; then

	curl -sL --retry 10 https://cmake.org/files/v3.12/${CMAKE}.tar.gz > ${CMAKE}.tar.gz
	sha512sum -c cmake.sha512

	tar xzf ${CMAKE}.tar.gz
fi
cd $CMAKE

sed -i '/"lib64"/s/64//' Modules/GNUInstallDirs.cmake &&

./bootstrap --prefix=/usr        \
            --mandir=/share/man  \
            --no-system-jsoncpp  \
            --no-system-librhash \
            --docdir=/share/doc/cmake-3.12.4 &&
make
make install
