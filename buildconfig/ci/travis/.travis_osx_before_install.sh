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

function clear_package_cache {
  rm -f "$HOME/HomebrewLocal/json/$1--*"
  if [[ -e "$HOME/HomebrewLocal/path/$1" ]]; then
    echo "Removing cached $1."
    rm -f $(< "$HOME/HomebrewLocal/path/$1") && rm -f "$HOME/HomebrewLocal/path/$1"
  fi
}

function install_or_upgrade {
  if [[ ! "$1" ]]; then
    echo "Called install_or_upgrade with no args; do nothing."
    return 0
  fi

  local outdated=$(brew outdated | grep -m 1 "$1")
  if [[ ! "$outdated" ]] && (brew ls --versions "$1" >/dev/null); then
    echo "$1 is already installed and up to date."
    return 0
  fi

  local deps=""
  local bottled=$(brew info "$1" | grep -m 1 "(bottled)")
  if [[ "$bottled" ]]; then
    deps=$(brew deps --1 "$1")
  else
    deps=$(brew deps --1 --include-build "$1")
  fi
  if [[ "$deps" ]]; then
    echo -n "$1 dependencies: "
    echo $deps
    while read -r dependency; do
      echo "$1: Install dependency $dependency."
      install_or_upgrade "$dependency" ${UNIVERSAL_FLAG}
    done <<< "$deps"
  fi

  if [[ "$outdated" ]]; then
    echo "$1 is installed but outdated."
    if [[ "$bottled" ]]; then
      echo "$1: Found bottle."
      clear_package_cache "$1"
      retry brew upgrade "$1"
      return 0
    else
      if [[ ! "$UPDATE_UNBOTTLED" ]] || [[ "$UPDATE_UNBOTTLED" = "0" ]]; then
        echo "$1: skipping update. (UPDATE_UNBOTTLED = 0)"
        return 0
      fi
      brew uninstall --ignore-dependencies "$1"
      clear_package_cache "$1"
    fi
  else
    echo "$1 is not installed."
    if [[ "$bottled" ]]; then
      echo "$1: Found bottle."
      clear_package_cache "$1"
      retry brew install "$@"
      return 0
    fi
  fi

  echo "$1: Found no bottle. Let's build one."

  retry brew install --build-bottle "$@"
  brew bottle --json "$1"
  # TODO: ^ first line in stdout is the bottle file
  # use instead of file cmd. json file has a similar name. "| head -n 1"?
  local jsonfile=$(find . -name $1*.bottle.json)
  brew uninstall --ignore-dependencies "$1"

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
    brew bottle --merge --write "$jsonfile"
  done
  echo "Done checking local bottles."
}

function prevent_stall {
    while kill -0 "$!" 2> /dev/null
    do
        sleep 20
        echo "Waiting..."
    done
}

check_local_bottles

if [ "${1}" == "--no-installs" ]; then
  unset HOMEBREW_BUILD_BOTTLE
  unset HOMEBREW_BOTTLE_ARCH
  return 0
fi

set +e

install_or_upgrade sdl ${UNIVERSAL_FLAG}
install_or_upgrade jpeg ${UNIVERSAL_FLAG}
install_or_upgrade libpng ${UNIVERSAL_FLAG}
install_or_upgrade libtiff ${UNIVERSAL_FLAG} --with-xz
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
brew tap pygame/portmidi
brew tap-pin pygame/portmidi
UPDATE_UNBOTTLED='1' install_or_upgrade cmake
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
