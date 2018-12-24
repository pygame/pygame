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
  local deps=""
  if (brew info "$1" | grep "(bottled)" >/dev/null); then
    deps=$(brew deps "$1")
  else
    deps=$(brew deps --include-build "$1")
  fi
  if [[ "$deps" ]]; then
    echo -n "$1 dependencies: "
    echo $deps
    while read -r dependency; do
      echo "$1: Install dependency $dependency."
      install_or_upgrade "$dependency"
    done <<< "$deps"
  fi

  if (brew ls --versions "$1" >/dev/null) && ! (brew outdated | grep "$1" >/dev/null); then
    echo "$1 is already installed and up to date."
  else
    if (brew outdated | grep "$1" >/dev/null); then
      echo "$1 is installed but outdated."
      if (brew info "$1" | grep "(bottled)" >/dev/null); then
        echo "$1: Found bottle."
        retry brew upgrade "$1"
        return 0
      else
        brew uninstall --ignore-dependencies "$1"
      fi
    else
      echo "$1 is not installed."
      if (brew info "$1" | grep "(bottled)" >/dev/null); then
        echo "$1: Found bottle."
        retry brew install "$1"
        return 0
      fi
    fi

    echo "$1: Found no bottle. Let's build one."

    retry brew install --build-bottle "$@"
    brew bottle --json "$@"
    # TODO: ^ first line in stdout is the bottle file
    # use instead of file cmd. json file has a similar name. "| head -n 1"?
    local jsonfile=$(find . -name $1*.bottle.json)
    brew uninstall --ignore-dependencies "$@"

    local bottlefile=$(find . -name $1*.tar.gz)
    echo "brew install $bottlefile"
    brew install "$bottlefile"

    # Add the bottle info into the package's formula
    echo "brew bottle --merge --write $jsonfile"
    brew bottle --merge --write "$jsonfile"

    # Path to the cachefile will be updated now
    local cachefile=$(brew --cache $1)
    echo "Copying $bottlefile to $cachefile..."
    cp -f "$bottlefile" "$cachefile"

    # save bottle info
    echo "Copying $jsonfile to $HOME/HomebrewLocal/json..."
    mkdir -p "$HOME/HomebrewLocal/json"
    cp -f "$jsonfile" "$HOME/HomebrewLocal/json/"

    echo "Saving bottle path to to $HOME/HomebrewLocal/path/$1..."
    mkdir -p "$HOME/HomebrewLocal/path"
    echo "$cachefile" > "$HOME/HomebrewLocal/path/$1"
    echo "Result: $(cat $HOME/HomebrewLocal/path/$1)."
  fi
}

function prevent_stall {
    while kill -0 "$!" 2> /dev/null
    do
        sleep 20
        echo "Waiting..."
    done
}

function check_local_bottles {
  echo "Checking local bottles in $HOME/HomebrewLocal/json/..."
  for jsonfile in $HOME/HomebrewLocal/json/*.json; do
    [ -e "$jsonfile" ] || continue
    local pkg="$(cut -d'-' -f1 <<<"$(basename $jsonfile)")"
    echo "Package: $pkg. JSON: $jsonfile."

    local filefull=$(cat $HOME/HomebrewLocal/path/$pkg)
    local file=$(basename $filefull)
    echo "$pkg: local bottle path: $filefull"

    # This might be good enough for now?
    echo "Adding local bottle into $pkg's formula."
    brew bottle --merge --write "$jsonfile"

    # TODO: check if the local bottle is still appropriate (by comparing versions and rebuild numbers)
    # if it does, re-add bottle info to formula like above
    # if it doesn't, delete cached bottle & json
    #    ie rm -f $filefull
    #brew info --json=v1 "$pkg"
    #brew info --json=v1 "$filefull"
  done
  echo "Done checking local bottles."
}

check_local_bottles


set +e

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

set -e

# brew install https://gist.githubusercontent.com/illume/08f9d3ca872dc2b61d80f665602233fd/raw/0fbfd6657da24c419d23a6678b5715a18cd6560a/portmidi.rb

unset HOMEBREW_BUILD_BOTTLE
unset HOMEBREW_BOTTLE_ARCH


echo "finished buildconfig/ci/travis/.travis_osx_before_install.sh"
