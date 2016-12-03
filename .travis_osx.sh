#!/bin/bash

set -e
git clone https://github.com/MacPython/terryfy.git
cd terryfy
# Work with a specific commit
git checkout 63fab201fab1f42ad213221be384fdd52ddf6561
cd ..
source terryfy/travis_tools.sh
get_python_environment homebrew $PY_VERSION $(pwd)/_test_env
