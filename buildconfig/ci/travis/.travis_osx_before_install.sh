# Compiles pygame on homebrew for distribution.
# This may not be what you want to do if not on travisci.

set -e

# Work around https://github.com/travis-ci/travis-ci/issues/8703 :-@
# Travis overrides cd to do something with Ruby. Revert to the default.
unset -f cd
shell_session_update() { :; }


set UPDATE_UNBOTTLED='0'


echo -en 'travis_fold:start:brew.update\\r'
echo "Updating Homebrew listings..."
brew update
echo -en 'travis_fold:end:brew.update\\r'
export HOMEBREW_NO_AUTO_UPDATE=1

brew install ccache
export PATH="/usr/local/opt/ccache/libexec:$PATH"

brew uninstall gdbm --ignore-dependencies
brew uninstall sqlite --ignore-dependencies
brew uninstall openssl --ignore-dependencies
brew uninstall readline --ignore-dependencies

brew uninstall --force --ignore-dependencies pkg-config
brew install pkg-config

if [[ ${BUILD_UNIVERSAL} == "1" ]]; then
  UNIVERSAL_FLAG='--universal'
  echo "Using --universal option for builds"
else
  UNIVERSAL_FLAG=''
  echo "Not using --universal option for builds"
fi

# Only compile from source if doing a release. on tag or master.
# This saves compile times for normal PR testing.
echo "About to install dependencies"
echo $TRAVIS_TAG
echo $TRAVIS_BRANCH
echo $TRAVIS_PULL_REQUEST
if [ "$TRAVIS_PULL_REQUEST" = "false" ] && ([ -n "$TRAVIS_TAG" ] || [ "$TRAVIS_BRANCH" = "master" ]); then
	echo "building more things from source"

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
fi


source "buildconfig/ci/travis/.travis_osx_utils.sh"

function clear_package_cache {
  rm -f "$HOME/HomebrewLocal/json/$1--*"
  if [[ -e "$HOME/HomebrewLocal/path/$1" ]]; then
    echo "Removing cached $1."
    rm -f $(< "$HOME/HomebrewLocal/path/$1") && rm -f "$HOME/HomebrewLocal/path/$1"
  fi
}

function check_local_bottles {
  echo "Checking local bottles in $HOME/HomebrewLocal/json/..."
  for jsonfile in $HOME/HomebrewLocal/json/*.json; do
    [ -e "$jsonfile" ] || continue
    local pkg="$(sed 's/\(.*\)--.*/\1/' <<<"$(basename $jsonfile)")"
    echo "Package: $pkg. JSON: $jsonfile."

    local filefull=$(< "$HOME/HomebrewLocal/path/$pkg")
    local file=$(basename $filefull)
    echo "$pkg: local bottle path: $filefull"

    echo "Adding local bottle into $pkg's formula."
    brew bottle --merge --write "$jsonfile" || true
  done
  echo "Done checking local bottles."
}

check_local_bottles

if [ "${1}" == "--no-installs" ]; then
  unset HOMEBREW_BUILD_BOTTLE
  unset HOMEBREW_BOTTLE_ARCH
  return 0
fi

set +e

brew tap pygame/portmidi
brew tap-pin pygame/portmidi

install_or_upgrade sdl ${UNIVERSAL_FLAG}
install_or_upgrade jpeg ${UNIVERSAL_FLAG}
UPDATE_UNBOTTLED='1' install_or_upgrade libpng ${UNIVERSAL_FLAG}
UPDATE_UNBOTTLED='1' install_or_upgrade xz ${UNIVERSAL_FLAG}
UPDATE_UNBOTTLED='1' install_or_upgrade libtiff ${UNIVERSAL_FLAG}
install_or_upgrade webp ${UNIVERSAL_FLAG}
install_or_upgrade libogg ${UNIVERSAL_FLAG}
install_or_upgrade libvorbis ${UNIVERSAL_FLAG}
install_or_upgrade flac ${UNIVERSAL_FLAG}
install_or_upgrade boost & prevent_stall #workaround due to glib
install_or_upgrade fluid-synth
install_or_upgrade libmikmod ${UNIVERSAL_FLAG}
install_or_upgrade smpeg


# Because portmidi hates us... and installs python2, which messes homebrew up.
# So we install portmidi from our own formula.
install_or_upgrade portmidi ${UNIVERSAL_FLAG}

install_or_upgrade freetype ${UNIVERSAL_FLAG}
install_or_upgrade sdl_ttf ${UNIVERSAL_FLAG}
install_or_upgrade sdl_image ${UNIVERSAL_FLAG}
install_or_upgrade sdl_mixer ${UNIVERSAL_FLAG} --with-flac --with-fluid-synth --with-libmikmod --with-libvorbis --with-smpeg

set -e

# brew install https://gist.githubusercontent.com/illume/08f9d3ca872dc2b61d80f665602233fd/raw/0fbfd6657da24c419d23a6678b5715a18cd6560a/portmidi.rb

unset HOMEBREW_BUILD_BOTTLE
unset HOMEBREW_BOTTLE_ARCH


echo "finished buildconfig/ci/travis/.travis_osx_before_install.sh"
