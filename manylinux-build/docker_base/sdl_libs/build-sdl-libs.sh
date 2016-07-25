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
./configure --enable-png --disable-png-shared --enable-jpg --disable-jpg-shared
make
make install
cd ..

# Build SDL_image
tar xzf ${IMG}.tar.gz
cd $IMG
./configure --enable-png --disable-png-shared --enable-jpg --disable-jpg-shared
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
./configure
make
make install
cd ..
