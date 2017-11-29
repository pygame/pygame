#!/bin/bash

set -ex

# Circumvent https://github.com/direnv/direnv/issues/210
shell_session_update() { :; }

git clone https://github.com/MacPython/terryfy.git
# Work around https://github.com/travis-ci/travis-ci/issues/8703 :-@
set +e
cd terryfy
# Work with a specific commit
git checkout 703737bd7be3a5d388146d5a95241ec2a17a4b2c
cd ..
source terryfy/travis_tools.sh
set -e

get_python_environment homebrew $PY_VERSION $(pwd)/_test_env
