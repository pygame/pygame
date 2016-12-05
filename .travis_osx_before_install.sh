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

brew uninstall --force --ignore-dependencies sdl
brew uninstall --force --ignore-dependencies sdl_image
brew uninstall --force --ignore-dependencies sdl_mixer
brew uninstall --force --ignore-dependencies sdl_ttf
brew uninstall --force --ignore-dependencies smpeg
brew uninstall --force --ignore-dependencies jpeg
brew uninstall --force --ignore-dependencies libpng
brew uninstall --force --ignore-dependencies libtiff
brew uninstall --force --ignore-dependencies webp
brew uninstall --force --ignore-dependencies flac
brew uninstall --force --ignore-dependencies fluid-synth
brew uninstall --force --ignore-dependencies libmikmod
brew uninstall --force --ignore-dependencies libvorbis
brew uninstall --force --ignore-dependencies smpeg
brew uninstall --force --ignore-dependencies portmidi
brew uninstall --force --ignore-dependencies freetype

brew install sdl ${UNIVERSAL_FLAG}
brew install jpeg ${UNIVERSAL_FLAG}
brew install libpng ${UNIVERSAL_FLAG}
brew install libtiff ${UNIVERSAL_FLAG} --with-xz
brew install webp ${UNIVERSAL_FLAG} --with-libtiff --with-giflib
brew install libogg ${UNIVERSAL_FLAG}
brew install libvorbis ${UNIVERSAL_FLAG}
brew install flac ${UNIVERSAL_FLAG} --with-libogg
brew install fluid-synth
brew install libmikmod ${UNIVERSAL_FLAG}
brew install smpeg
brew install portmidi ${UNIVERSAL_FLAG}
brew install freetype ${UNIVERSAL_FLAG}
brew install sdl_ttf ${UNIVERSAL_FLAG}
brew install sdl_image ${UNIVERSAL_FLAG}
brew install sdl_mixer ${UNIVERSAL_FLAG} --with-flac --with-fluid-synth --with-libmikmod --with-libvorbis --with-smpeg
