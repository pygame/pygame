#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

GETTEXT=gettext-0.21

curl -sL --retry 10 https://ftp.gnu.org/gnu/gettext/${GETTEXT}.tar.gz > ${GETTEXT}.tar.gz
sha512sum -c gettext.sha512

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
      # linux
      export GETTEXT_CONFIGURE=
elif [[ "$OSTYPE" == "darwin"* ]]; then
      # Mac OSX, ship libintl.h on mac.
      export GETTEXT_CONFIGURE=--with-included-gettext
fi

tar xzf ${GETTEXT}.tar.gz
cd $GETTEXT

./configure $ARCHS_CONFIG_FLAG $GETTEXT_CONFIGURE  \
--disable-dependency-tracking \
--disable-silent-rules \
--disable-debug \
--with-included-glib \
--with-included-libcroco \
--with-included-libunistring \
--with-included-libxml \
--without-emacs \
--disable-java \
--disable-csharp \
--without-git \
--without-cvs \
--without-xz

make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi
