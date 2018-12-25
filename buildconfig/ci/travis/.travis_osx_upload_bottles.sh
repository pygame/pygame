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

if [[ ${BUILD_UNIVERSAL} == "1" ]]; then
  UNIVERSAL_FLAG='--universal'
  echo "Using --universal option for builds"
else
  UNIVERSAL_FLAG=''
  echo "Not using --universal option for builds"
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

set shadowed_pkgs
shadowed_pkgs=()

function not_shadowed {
  for pkg in $shadowed_pkgs; do
    if [[ "$pkg" = "$1" ]]; then
      return 1
    fi
  done
  return 0
}

function upload_bottle {
  if [[ ! "$1" ]]; then
    echo "Called upload_bottle with no args; do nothing."
    return 0
  fi

  local outdated=$(brew outdated | grep -m 1 "$1")
  if [[ ! "$outdated" ]] && (brew ls --versions "$1" >/dev/null); then
    echo "$1 is already installed and up to date."
    return 0
  fi

  local deps=$(brew deps --1 "$1")
  if [[ "$deps" ]]; then
    echo -n "$1 dependencies: "
    echo $deps
    while read -r dependency; do
      echo "$1: Dependency $dependency."
      upload_bottle "$dependency"
    done <<< "$deps"
  fi

  local bottled=$(brew info "$1" | grep -m 1 "(bottled)")
  if [[ "$outdated" ]]; then
    echo "$1 is installed but outdated."
    if [[ "$bottled" ]]; then
      if (not_shadowed "$1"); then
        echo "$1: Found bottle. Skipping."
        return 0
      fi
    fi
    brew uninstall --ignore-dependencies "$@"
  else
    echo "$1 is not installed."
    if [[ "$bottled" ]]; then
      if (not_shadowed "$1"); then
        echo "$1: Found bottle. Skipping."
        return 0
      fi
    fi
  fi

  echo "Found no bottle for $1. Let's build one."

  retry brew install --build-bottle "$@"
  brew bottle --json "$@"
  # TODO: ^ first line in stdout is the bottle file
  # use instead of file cmd. json file has a similar name. "| head -n 1"?
  local jsonfile=$(find . -name $1*.bottle.json)
  brew uninstall --ignore-dependencies "$@"
  local bottlefile=$(find . -name $1*.tar.gz)
  brew install "$bottlefile" # can this be removed?

  # Add the bottle info into the package's formula
  echo "brew bottle --merge --write $jsonfile"
  brew bottle --merge --write "$jsonfile"

  # Path to the cachefile will be updated now
  #local cachefile=$(brew --cache $1)

  mkdir -p Formula
  brew cat $1 > "Formula/$1.rb"
  mkdir -p "bottles"
  mv "$bottlefile" "bottles/"
  #tar cfzv "$1-formula.tar.gz" "$1.rb" "$bottlefile"

  # upload the package
  #curl -T "$1-formula.tar.gz" "https://transfer.sh/$archive"
}

function add_shadowed {
  if [[ ! "$1" ]]; then
    return 0
  fi
  shadowed_pkgs+=("$1")
}

function prevent_stall {
    while kill -0 "$!" 2> /dev/null
    do
        sleep 5
        echo "Waiting..."
    done
}

set +e
brew tap pygame/portmidi
# Extract packages to build from these strings: "Warning: python is provided by core, but is now shadowed by pygame/portmidi/python."
#   ie "python" in the above example
brew tap-pin pygame/portmidi | sed 's/^Warning: \(\w\+\).*/\1/g' | add_shadowed
echo "Shadowed packages: ${shadowed_pkgs[@]}"

IFS=';' pkgs=( "$BOTTLES_BUILD" )
for pkg in $pkgs; do
  upload_bottle "$pkg" & prevent_stall
done

# archive all packages upload the archive
echo "Creating brew-packages.tar.gz..."
tar cfzv "brew-packages.tar.gz" "Formula" "bottles"

set -e

echo "Uploading brew-packages.tar.gz..."
curl -vs --upload-file "./brew-packages.tar.gz" "https://transfer.sh/" &> /dev/stdout
echo "

"
sleep 5
