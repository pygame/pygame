# This uses manylinux build scripts to build dependencies
#  on mac.
#
# Warning: this should probably not be run on your own mac.
#   Since it will install all these deps all over the place,
#   and they may conflict with existing installs you have.

set -e -x
cd ../manylinux-build/docker_base

# to use the gnu readlink, needs `brew install coreutils`
export PATH="/usr/local/opt/coreutils/libexec/gnubin:$PATH"

# for great speed.
export MAKEFLAGS="-j 4"
export MACOSX_DEPLOYMENT_TARGET=10.9

# First clean up some homebrew stuff we don't want linked in.
rm -rf /usr/local/bin/curl
rm -rf /usr/local/opt/curl
rm -rf /usr/local/bin/git
rm -rf /usr/local/opt/git
# ln -s /usr/bin/curl /usr/local/bin/curl
ln -s /usr/bin/git /usr/local/bin/git

rm -rf /usr/local/lib/libtiff*
rm -rf /usr/local/lib/libsndfile*
rm -rf /usr/local/lib/glib*
rm -rf /usr/local/lib/libglib*
rm -rf /usr/local/lib/libgthread*
rm -rf /usr/local/lib/libintl*
rm -rf /usr/local/lib/libbrotlidec*
rm -rf /usr/local/lib/libopus*

rm -rf /usr/local/opt/gettext
rm -rf /usr/local/Cellar/libsndfile
rm -rf /usr/local/Cellar/glib
rm -rf /usr/local/Cellar/brotli
rm -rf /usr/local/Cellar/glib
rm -rf /usr/local/Cellar/libtiff

rm -rf /usr/local/share/doc/tiff-*
rm -rf /usr/local/share/doc/libsndfile


# bash glib/build-glib.sh
bash zlib/build-zlib.sh
bash libpng/build-png.sh
bash libjpeg/build-jpeg.sh
bash libtiff/build-tiff.sh
bash libmodplug/build-libmodplug.sh
bash ogg/build-ogg.sh
bash flac/build-flac.sh
bash libwebp/build-webp.sh
bash freetype/build-freetype.sh
bash mpg123/build-mpg123.sh
bash sndfile/build-sndfile.sh
# bash fluidsynth/build-fluidsynth.sh
bash sdl_libs/build-sdl2-libs.sh

bash harfbuzz/build-harfbuzz.sh

bash portmidi/build-portmidi.sh
# strangely somehow the built pygame links against the libportmidi.dylib here:
cp /usr/local/lib/libportmidi.dylib /Users/runner/work/pygame/pygame/libportmidi.dylib
