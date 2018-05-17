# Compiles pygame on homebrew for distribution.
# This may not be what you want to do if not on travisci.

set -e

# Work around https://github.com/travis-ci/travis-ci/issues/8703 :-@
# Travis overrides cd to do something with Ruby. Revert to the default.
unset -f cd
shell_session_update() { :; }



echo -en 'travis_fold:start:brew.update\\r'
echo "Updating Homebrew listings..."
brew update
echo -en 'travis_fold:end:brew.update\\r'
export HOMEBREW_NO_AUTO_UPDATE=1

brew uninstall --force --ignore-dependencies pkg-config
brew install pkg-config

# brew install ccache
# export PATH=/usr/local/opt/ccache/libexec:$PATH

if [[ ${BUILD_UNIVERSAL} == "1" ]]; then
  UNIVERSAL_FLAG='--universal'
  echo "Using --universal option for builds"
else
  UNIVERSAL_FLAG=''
  echo "Not using --universal option for builds"
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

# seems portmidi added python in homebrew, which is broken with our travis python builds on mac.
#brew install portmidi ${UNIVERSAL_FLAG}

brew install https://gist.githubusercontent.com/illume/08f9d3ca872dc2b61d80f665602233fd/raw/0fbfd6657da24c419d23a6678b5715a18cd6560a/portmidi.rb

brew install freetype ${UNIVERSAL_FLAG}
brew install sdl_ttf ${UNIVERSAL_FLAG}
brew install sdl_image ${UNIVERSAL_FLAG}
brew install sdl_mixer ${UNIVERSAL_FLAG} --with-flac --with-fluid-synth --with-libmikmod --with-libvorbis --with-smpeg

echo "finished .travis_osx_before_install.sh"
