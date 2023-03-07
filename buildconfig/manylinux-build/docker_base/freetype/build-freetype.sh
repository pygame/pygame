#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

FREETYPE=freetype-2.12.1
HARFBUZZ_VER=5.1.0
HARFBUZZ_NAME="harfbuzz-$HARFBUZZ_VER"

curl -sL --retry 10 http://download.savannah.gnu.org/releases/freetype/${FREETYPE}.tar.gz > ${FREETYPE}.tar.gz
curl -sL --retry 10 https://github.com/harfbuzz/harfbuzz/releases/download/${HARFBUZZ_VER}/${HARFBUZZ_NAME}.tar.xz > ${HARFBUZZ_NAME}.tar.xz
sha512sum -c freetype.sha512

# extract installed sources
tar xzf ${FREETYPE}.tar.gz
unxz ${HARFBUZZ_NAME}.tar.xz
tar xf ${HARFBUZZ_NAME}.tar

# freetype and harfbuzz have an infamous circular dependency, which is why
# this file is not like the rest of docker_base files

# 1. First compile freetype without harfbuzz support
cd $FREETYPE

./configure $ARCHS_CONFIG_FLAG --with-harfbuzz=no
make
make install  # this freetype is not installed to mac cache dir


# harfbuzz was not well tested, only enable on linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then

    cd ..

    # 2. Compile harfbuzz with freetype support
    cd ${HARFBUZZ_NAME}

    # harfbuzz has a load of optional dependencies but only freetype is important
    # to us.
    # Cairo and chafa are only needed for harfbuzz commandline utilities so we
    # don't use it. glib available is a bit old so we don't prefer it as of now.
    # we also don't compile-in icu so that harfbuzz uses built-in unicode handling
    # LDFLAGS are passed explicitly so that harfbuzz picks the freetype we
    # installed first
    ./configure $ARCHS_CONFIG_FLAG --with-freetype=yes \
        --with-cairo=no --with-chafa=no --with-glib=no --with-icu=no \
        --disable-static LDFLAGS="-L/usr/local/lib"
    make
    make install

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Install to mac deps cache dir as well
        make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
    fi

    cd ..

    # 3. Recompile freetype, and this time with harfbuzz support
    cd $FREETYPE

    # fully clean previous install
    make clean

    ./configure $ARCHS_CONFIG_FLAG --with-harfbuzz=yes
    make
    make install

fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
