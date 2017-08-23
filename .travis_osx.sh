#!/bin/bash

set -e
git clone https://github.com/MacPython/terryfy.git
cd terryfy
# Work with a specific commit
git checkout 07480be3e0b3490495cb8a9629e55be54c3adac3
cd ..
source terryfy/travis_tools.sh
get_python_environment homebrew $PY_VERSION $(pwd)/_test_env
