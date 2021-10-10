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


# First clean up some homebrew stuff we don't want linked in
# ----------------------------------------------------------

rm -rf /usr/local/bin/curl
rm -rf /usr/local/opt/curl
rm -rf /usr/local/bin/git
rm -rf /usr/local/opt/git
# Use the apple provided curl, and git.
#     The homebrew ones depend on libs we don't want to include.
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
rm -rf /usr/local/opt/freetype

rm -rf /usr/local/Cellar/libtiff
rm -rf /usr/local/Cellar/libsndfile
rm -rf /usr/local/Cellar/glib
rm -rf /usr/local/Cellar/brotli
rm -rf /usr/local/Cellar/pcre
rm -rf /usr/local/Cellar/opus
rm -rf /usr/local/Cellar/freetype

rm -rf /usr/local/opt/gettext

rm -rf /usr/local/share/doc/tiff-*
rm -rf /usr/local/share/doc/libsndfile
rm -rf /usr/local/share/glib-2.0
rm -rf /usr/local/share/gdb/auto-load

rm -rf /usr/local/include/glib-2.0
rm -rf /usr/local/include/gio-unix-2.0


# Now start installing dependencies
# ---------------------------------

# sdl_image deps
bash zlib/build-zlib.sh
bash libpng/build-png.sh # depends on zlib
bash libjpeg/build-jpeg.sh
bash libtiff/build-tiff.sh
bash libwebp/build-webp.sh

# sdl_ttf deps
# export EXTRA_CONFIG_FREETYPE=--without-harfbuzz
# bash freetype/build-freetype.sh
# bash harfbuzz/build-harfbuzz.sh
# export EXTRA_CONFIG_FREETYPE=
bash freetype/build-freetype.sh

# sdl_mixer deps
bash libmodplug/build-libmodplug.sh
bash ogg/build-ogg.sh
bash flac/build-flac.sh
bash mpg123/build-mpg123.sh

# fluidsynth (for sdl_mixer)
bash gettext/build-gettext.sh
bash glib/build-glib.sh # depends on gettext
bash sndfile/build-sndfile.sh
sudo mkdir -p /usr/local/lib64 # the install tries to put something in here
sudo bash fluidsynth/build-fluidsynth.sh # sudo otherwise install doesn't work.

bash sdl_libs/build-sdl2-libs.sh

# for pygame.midi
bash portmidi/build-portmidi.sh
# strangely somehow the built pygame links against the libportmidi.dylib here:
cp /usr/local/lib/libportmidi.dylib /Users/runner/work/pygame/pygame/libportmidi.dylib
