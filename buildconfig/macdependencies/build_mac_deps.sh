# This uses manylinux build scripts to build dependencies
#  on mac.
#
# Warning: this should probably not be run on your own mac.
#   Since it will install all these deps all over the place,
#   and they may conflict with existing installs you have.

set -e -x

export MACDEP_CACHE_PREFIX_PATH=${GITHUB_WORKSPACE}/pygame_mac_deps_${MAC_ARCH}

bash ./clean_usr_local.sh
mkdir $MACDEP_CACHE_PREFIX_PATH

# to use the gnu readlink, needs `brew install coreutils`
export PATH="/usr/local/opt/coreutils/libexec/gnubin:$PATH"

# for great speed.
export MAKEFLAGS="-j 4"

# With this we
# 1) Force install prefix to /usr/local
# 2) use lib directory within /usr/local (and not lib64)
# 3) make release binaries
# 4) build shared libraries
# 5) not have @rpath in the linked dylibs (needed on macs only)
export PG_BASE_CMAKE_FLAGS="-DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DCMAKE_INSTALL_LIBDIR:PATH=lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=true \
    -DCMAKE_INSTALL_NAME_DIR=/usr/local/lib"

if [[ "$MAC_ARCH" == "arm64" ]]; then
    # for scripts using ./configure to make arm64 binaries
    export CC="clang -target arm64-apple-macos11.0"
    export CXX="clang++ -target arm64-apple-macos11.0"

    # This does not do anything actually, but without this ./configure errors
    export ARCHS_CONFIG_FLAG="--host=aarch64-apple-darwin20.0.0"
    
    # configure cmake to cross-compile
    export PG_BASE_CMAKE_FLAGS="$PG_BASE_CMAKE_FLAGS -DCMAKE_OSX_ARCHITECTURES=arm64"

    # we don't need mac 10.9 support while compiling for apple M1 macs
    export MACOSX_DEPLOYMENT_TARGET=11.0
else
    # install NASM to generate optimised x86_64 libjpegturbo builds
    brew install nasm

    export MACOSX_DEPLOYMENT_TARGET=10.9
fi

cd ../manylinux-build/docker_base

# Now start installing dependencies
# ---------------------------------

sudo mkdir -p /usr/local/man/man1  # the install tries to put something in here
sudo chmod 0777 /usr/local/man/man1  # so that install can put files here
mkdir -p ${MACDEP_CACHE_PREFIX_PATH}/usr/local/man/man1

# sdl_image deps
bash zlib-ng/build-zlib-ng.sh
bash libpng/build-png.sh # depends on zlib
bash libjpegturbo/build-jpeg-turbo.sh
bash libtiff/build-tiff.sh
bash libwebp/build-webp.sh

# freetype (also sdl_ttf dep)
bash brotli/build-brotli.sh
bash bzip2/build-bzip2.sh
bash freetype/build-freetype.sh

# sdl_mixer deps
bash libmodplug/build-libmodplug.sh
bash ogg/build-ogg.sh
bash flac/build-flac.sh
bash mpg123/build-mpg123.sh
bash opus/build-opus.sh # needs libogg (which is a container format)

# fluidsynth (for sdl_mixer)
bash gettext/build-gettext.sh
bash glib/build-glib.sh # depends on gettext
bash sndfile/build-sndfile.sh
bash fluidsynth/build-fluidsynth.sh

bash sdl_libs/build-sdl2-libs.sh

# for pygame.midi
bash portmidi/build-portmidi.sh
