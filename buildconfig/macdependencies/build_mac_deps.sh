# This uses manylinux build scripts to build dependencies
#  on mac.
#
# Warning: this should probably not be run on your own mac.
#   Since it will install all these deps all over the place,
#   and they may conflict with existing installs you have.

set -e -x

export MACDEP_CACHE_PREFIX_PATH=${GITHUB_WORKSPACE}/pygame_mac_deps

bash ./clean_usr_local.sh
mkdir $MACDEP_CACHE_PREFIX_PATH

# to use the gnu readlink, needs `brew install coreutils`
export PATH="/usr/local/opt/coreutils/libexec/gnubin:$PATH"

# for great speed.
export MAKEFLAGS="-j 4"
export MACOSX_DEPLOYMENT_TARGET=10.9

cd ../manylinux-build/docker_base

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
sudo mkdir -p ${MACDEP_CACHE_PREFIX_PATH}/usr/local/lib64
sudo bash fluidsynth/build-fluidsynth.sh # sudo otherwise install doesn't work.

bash sdl_libs/build-sdl2-libs.sh

# for pygame.midi
bash portmidi/build-portmidi.sh
