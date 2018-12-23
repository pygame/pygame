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

brew install ccache
export PATH="/usr/local/opt/ccache/libexec:$PATH"

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
ccache -s
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

	# These are for building from source, with 'core2'
	#   because otherwise homebrew will use the architecture of the build host.
	export HOMEBREW_BUILD_BOTTLE=1
	export HOMEBREW_BOTTLE_ARCH=core2
fi


function fail {
  echo $1 >&2
  exit 1
}

function retry {
  local n=1
  local max=5
  local delay=2
  while true; do
    "$@" && break || {
      if [[ $n -lt $max ]]; then
        ((n++))
        echo "Command failed. Attempt $n/$max:"
        sleep $delay;
      else
        fail "The command has failed after $n attempts."
      fi
    }
  done
}

function install_or_upgrade {
	set +e
    if brew ls --versions "$1" >/dev/null; then
        echo "package already installed: $@"
    else
    	retry brew install "$@"
    fi
    set -e
}

function prevent_stall {
    while kill -0 "$!" 2> /dev/null
    do
        sleep 20
        echo "Waiting..."
    done
}


install_or_upgrade sdl ${UNIVERSAL_FLAG}
install_or_upgrade jpeg ${UNIVERSAL_FLAG}
install_or_upgrade libpng ${UNIVERSAL_FLAG}
install_or_upgrade libtiff ${UNIVERSAL_FLAG} --with-xz
install_or_upgrade webp ${UNIVERSAL_FLAG}
install_or_upgrade libogg ${UNIVERSAL_FLAG}
install_or_upgrade libvorbis ${UNIVERSAL_FLAG}
install_or_upgrade flac ${UNIVERSAL_FLAG}
brew upgrade boost & prevent_stall
install_or_upgrade fluid-synth
install_or_upgrade libmikmod ${UNIVERSAL_FLAG}
install_or_upgrade smpeg


# Because portmidi hates us... and installs python2, which messes homebrew up.
# So we install portmidi from our own formula.
brew tap pygame/portmidi
brew install pygame/portmidi/portmidi ${UNIVERSAL_FLAG}

install_or_upgrade freetype ${UNIVERSAL_FLAG}
install_or_upgrade sdl_ttf ${UNIVERSAL_FLAG}
install_or_upgrade sdl_image ${UNIVERSAL_FLAG}
install_or_upgrade sdl_mixer ${UNIVERSAL_FLAG} --with-flac --with-fluid-synth --with-libmikmod --with-libvorbis --with-smpeg

# brew install https://gist.githubusercontent.com/illume/08f9d3ca872dc2b61d80f665602233fd/raw/0fbfd6657da24c419d23a6678b5715a18cd6560a/portmidi.rb

unset HOMEBREW_BUILD_BOTTLE
unset HOMEBREW_BOTTLE_ARCH


echo "finished buildconfig/ci/travis/.travis_osx_before_install.sh"
