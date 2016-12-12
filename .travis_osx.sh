#!/bin/bash

set -e
git clone https://github.com/MacPython/terryfy.git
cd terryfy
# Work with a specific commit
git checkout b48eff2ea5f194d404e88235cec7f270a8e5f24f
cd ..
source terryfy/travis_tools.sh
get_python_environment homebrew $PY_VERSION $(pwd)/_test_env
