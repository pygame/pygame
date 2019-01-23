#!/bin/bash

# Order the osx scripts are run.
	# .travis_osx_before_install.sh
	# .travis_osx.sh
	# .travis_osx_install.sh
	# .travis_osx_after_success.sh
	# .travis_osx_rename_whl.py
	# .travis_osx_upload_whl.py --no-config

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


git clone https://github.com/illume/terryfy.git
# git clone https://github.com/MacPython/terryfy.git
cd terryfy
# Work with a specific commit
#git checkout 703737bd7be3a5d388146d5a95241ec2a17a4b2c
cd ..
source terryfy/travis_tools.sh

# Ensure that 'python' is on $PATH
if [[ "$PY_VERSION" == "2" ]]; then
    export PATH="/usr/local/opt/python/libexec/bin:$PATH"
fi

# try and install an old python3.6 formula
if [[ "$PY_VERSION_" == "3.6" ]]; then
	brew uninstall python --force --ignore-dependencies
	retry install_or_upgrade "https://raw.githubusercontent.com/pygame/homebrew-portmidi/master/Formula/python36/python.rb"
	export PYTHON_EXE=python3.6
	export PIP_CMD="python3.6 -m pip"
elif [[ "$PY_VERSION_" == "3.7" ]]; then
	brew uninstall python --force --ignore-dependencies
	retry install_or_upgrade python
	export PYTHON_EXE=python3.7
	export PIP_CMD="python3.7 -m pip"
else
	brew uninstall --force --ignore-dependencies python@2
	install_or_upgrade_deps "python@2"
	get_python_environment homebrew $PY_VERSION $(pwd)/_test_env
fi




fi
