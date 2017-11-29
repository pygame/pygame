#!/bin/bash

set -e

# Work around https://github.com/travis-ci/travis-ci/issues/8703 :-@
# Travis overrides cd to do something with Ruby. Revert to the default.
unset -f cd

git clone https://github.com/MacPython/terryfy.git
cd terryfy
# Work with a specific commit
git checkout 703737bd7be3a5d388146d5a95241ec2a17a4b2c
cd ..
source terryfy/travis_tools.sh

# Ensure that 'python' is on $PATH
if [[ "$PY_VERSION" == "2" ]]; then
    export PATH="/usr/local/opt/python/libexec/bin:$PATH"
fi

get_python_environment homebrew $PY_VERSION $(pwd)/_test_env
