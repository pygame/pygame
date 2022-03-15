#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

SDL2="SDL2-2.0.20"
IMG2="SDL2_image-2.0.5"
TTF2="SDL2_ttf-2.0.15"
MIX2="SDL2_mixer-2.0.4"


# Download
curl -sL --retry 10 https://www.libsdl.org/release/${SDL2}.tar.gz > ${SDL2}.tar.gz
# curl -sL --retry 10 https://www.libsdl.org/tmp/release/SDL2-2.0.14.tar.gz > SDL2-2.0.14.tar.gz
# curl -sL --retry 10 https://hg.libsdl.org/SDL/archive/tip.tar.gz > ${SDL2}.tar.gz

curl -sL --retry 10 https://www.libsdl.org/projects/SDL_image/release/${IMG2}.tar.gz > ${IMG2}.tar.gz
curl -sL --retry 10 https://www.libsdl.org/projects/SDL_ttf/release/${TTF2}.tar.gz > ${TTF2}.tar.gz
curl -sL --retry 10 https://www.libsdl.org/projects/SDL_mixer/release/${MIX2}.tar.gz > ${MIX2}.tar.gz
sha512sum -c sdl2.sha512



# Build SDL
tar xzf ${SDL2}.tar.gz

# this is for renaming the tip.tar.gz
# mv SDL-* ${SDL2}

if [[ "$MAC_ARCH" == "arm64" ]]; then
    # Build SDL with ARM optimisations on M1 macs
    export M1_MAC_EXTRA_FLAGS="--enable-arm-simd --enable-arm-neon"
fi

cd $SDL2
./configure --disable-video-vulkan $ARCHS_CONFIG_FLAG $M1_MAC_EXTRA_FLAGS
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi

cd ..


# Build SDL_image
tar xzf ${IMG2}.tar.gz
cd $IMG2
# The --disable-x-shared flags make it use standard dynamic linking rather than
# dlopen-ing the library itself. This is important for when auditwheel moves
# libraries into the wheel.
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
      # linux
      export SDL_IMAGE_CONFIGURE=
elif [[ "$OSTYPE" == "darwin"* ]]; then
      # Mac OSX
      # --disable-imageio is so it doesn't use the built in mac image loading.
      #     Since it is not as compatible with some jpg/png files.
      export SDL_IMAGE_CONFIGURE=--disable-imageio
fi

./configure --enable-png --disable-png-shared --enable-jpg --disable-jpg-shared \
        --enable-tif --disable-tif-shared --enable-webp --disable-webp-shared \
        $SDL_IMAGE_CONFIGURE $ARCHS_CONFIG_FLAG
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi

cd ..

# Build SDL_ttf
tar xzf ${TTF2}.tar.gz
cd $TTF2
./configure $ARCHS_CONFIG_FLAG
make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi

cd ..


# Build SDL_mixer
tar xzf ${MIX2}.tar.gz
cd $MIX2

# The --disable-x-shared flags make it use standard dynamic linking rather than
# dlopen-ing the library itself. This is important for when auditwheel moves
# libraries into the wheel.
./configure $ARCHS_CONFIG_FLAG \
      --disable-dependency-tracking \
      --disable-music-flac-shared \
      --disable-music-midi-fluidsynth-shared \
      --disable-music-mod-mikmod-shared \
      --disable-music-mod-modplug-shared \
      --disable-music-mp3-mpg123-shared \
      --disable-music-ogg-shared \
      --disable-music-opus-shared \
      --enable-music-midi-fluidsynth \
      --enable-music-mod-mikmod \
      --enable-music-mod-modplug \
      --enable-music-ogg \
      --enable-music-opus \
      --enable-music-flac \
      --enable-music-mp3-mpg123 \
      --enable-music-mp3 \
      --enable-music-mod

make
make install

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install to mac deps cache dir as well
    make install DESTDIR=${MACDEP_CACHE_PREFIX_PATH}
fi

cd ..
