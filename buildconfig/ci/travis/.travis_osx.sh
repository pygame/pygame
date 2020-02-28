#!/bin/bash

# Order the osx scripts are run.
	# .travis_osx_before_install.sh
	# .travis_osx.sh
	# .travis_osx_install.sh
	# .travis_osx_after_success.sh
	# .travis_osx_rename_whl.py

set -e

# Work around https://github.com/travis-ci/travis-ci/issues/8703 :-@
# Travis overrides cd to do something with Ruby. Revert to the default.
unset -f cd

if [[ "$PY_VERSION" == "pypy2" ]] || [[ "$PY_VERSION" == "pypy3" ]]; then
	brew install $PY_VERSION
	export PYTHON_EXE=$PY_VERSION
	export PIP_CMD="$PY_VERSION -m pip"
else

export HOMEBREW_NO_AUTO_UPDATE=1
source "buildconfig/ci/travis/.travis_osx_utils.sh"

# Ensure that 'python' is on $PATH
if [[ "$PY_VERSION" == "2" ]]; then
    export PATH="/usr/local/opt/python/libexec/bin:$PATH"
fi

if [[ "$PY_VERSION_" == "3.6" ]]; then
	brew uninstall python --force --ignore-dependencies
	retry brew install "https://raw.githubusercontent.com/pygame/homebrew-portmidi/master/Formula/python36/python.rb"
	export PYTHON_EXE=python3.6
	export PIP_CMD="python3.6 -m pip"
elif [[ "$PY_VERSION_" == "3.7" ]]; then
	brew uninstall python --force --ignore-dependencies
	retry brew install "https://raw.githubusercontent.com/pygame/homebrew-portmidi/master/Formula/python37/python.rb"
	export PYTHON_EXE=python3.7
	export PIP_CMD="python3.7 -m pip"
elif [[ "$PY_VERSION_" == "3.8" ]]; then
	brew uninstall python --force --ignore-dependencies
	retry brew install "https://raw.githubusercontent.com/pygame/homebrew-portmidi/master/Formula/python38/python.rb"
	export PYTHON_EXE=python3.8
	export PIP_CMD="python3.8 -m pip"
else
	brew uninstall --force --ignore-dependencies python@2
	retry brew install "pygame/portmidi/python@2"
	export PYTHON_EXE=python2.7
	export PIP_CMD="python2.7 -m pip"
fi


fi



echo "$PYTHON_EXE --version"
$PYTHON_EXE --version
echo "ls -la /usr/local/bin/"
ls -la /usr/local/bin/
