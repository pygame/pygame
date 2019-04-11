#!/bin/bash
set -e -x

cd /sdl_build/

SDL="SDL-1.2.15"
IMG="SDL_image-1.2.12"
TTF="SDL_ttf-2.0.11"
MIX="SDL_mixer-1.2.12"

# Download
curl -sL https://www.libsdl.org/release/${SDL}.tar.gz > ${SDL}.tar.gz
curl -sL https://www.libsdl.org/projects/SDL_image/release/${IMG}.tar.gz > ${IMG}.tar.gz
curl -sL https://www.libsdl.org/projects/SDL_ttf/release/${TTF}.tar.gz > ${TTF}.tar.gz
curl -sL https://www.libsdl.org/projects/SDL_mixer/release/${MIX}.tar.gz > ${MIX}.tar.gz
sha512sum -c sdl.sha512

# Build SDL
tar xzf ${SDL}.tar.gz
cd $SDL
patch -p1 < ../libsdl-1.2-fix-compilation-libX11.patch
./autogen.sh
./configure --enable-png --disable-png-shared --enable-jpg --disable-jpg-shared
make
make install
cd ..

# Link sdl-config into /usr/bin so that smpeg-config can find it
ln -s /usr/local/bin/sdl-config /usr/bin/

# Build SDL_image
tar xzf ${IMG}.tar.gz
cd $IMG
# The --disable-x-shared flags make it use standard dynamic linking rather than
# dlopen-ing the library itself. This is important for when auditwheel moves
# libraries into the wheel.
./configure --enable-png --disable-png-shared --enable-jpg --disable-jpg-shared \
        --enable-tif --disable-tif-shared --enable-webp --disable-webp-shared
make
make install
cd ..

# Build SDL_ttf
tar xzf ${TTF}.tar.gz
cd $TTF
./configure
make
make install
cd ..

# Build SDL_mixer
tar xzf ${MIX}.tar.gz
cd $MIX
# The --disable-x-shared flags make it use standard dynamic linking rather than
# dlopen-ing the library itself. This is important for when auditwheel moves
# libraries into the wheel.
./configure --enable-music-mod --disable-music-mod-shared \
            --enable-music-fluidsynth --disable-music-fluidsynth-shared \
            --enable-music-ogg  --disable-music-ogg-shared \
            --enable-music-flac  --disable-music-flac-shared \
            --enable-music-mp3  --disable-music-mp3-shared
make
make install
cd ..
