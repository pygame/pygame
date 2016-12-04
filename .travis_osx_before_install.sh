# Compiles pygame on homebrew for distribution.
# This may not be what you want to do if not on travisci.

set -e

echo -en 'travis_fold:start:brew.update\\r'
brew update
echo -en 'travis_fold:end:brew.update\\r'
export HOMEBREW_NO_AUTO_UPDATE=1

brew unlink pkg-config
brew install pkg-config
brew link pkg-config
brew unlink libpng
brew unlink libtiff

# brew install ccache
# export PATH=/usr/local/opt/ccache/libexec:$PATH

if [[ ${BUILD_UNIVERSAL} == "1" ]]; then
  UNIVERSAL_FLAG='--universal'
else
  UNIVERSAL_FLAG=''
fi

brew uninstall --force sdl
brew uninstall --force sdl_image
brew uninstall --force sdl_mixer
brew uninstall --force sdl_ttf
brew uninstall --force smpeg
brew uninstall --force flac
brew uninstall --force fluid-synth
brew uninstall --force smpeg
brew uninstall --force libmikmod
brew uninstall --force libvorbis
brew uninstall --force portmidi
brew uninstall --force freetype
brew install sdl
brew install jpeg ${UNIVERSAL_FLAG}
brew install libpng ${UNIVERSAL_FLAG}
brew install libtiff ${UNIVERSAL_FLAG} --with-xz
brew install webp ${UNIVERSAL_FLAG} --with-libtiff --with-giflib
brew install libvorbis ${UNIVERSAL_FLAG}
brew install libogg ${UNIVERSAL_FLAG}
brew install flac ${UNIVERSAL_FLAG} --with-libogg
brew install fluid-synth
brew install libmikmod ${UNIVERSAL_FLAG}
brew install smpeg
brew install portmidi
brew install freetype ${UNIVERSAL_FLAG}
brew install sdl_ttf ${UNIVERSAL_FLAG}

brew install sdl_image ${UNIVERSAL_FLAG}
brew install sdl_mixer --with-flac --with-fluid-synth --with-libmikmod --with-libvorbis --with-smpeg
