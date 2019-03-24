#!/bin/bash
# set -e -x

# cd $(dirname `readlink -f "$0"`)

PNG=libpng-1.6.36

curl -sL http://download.sourceforge.net/libpng/${PNG}.tar.gz > ${PNG}.tar.gz
# sha512sum -c png.sha512

# tar xzf ${PNG}.tar.gz
# cd $PNG
# ./configure
# make
# make install
