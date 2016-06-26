#!/bin/bash
set -e

git clone https://github.com/MacPython/terryfy.git
cd terryfy
# Work with a specific commit
git checkout c9800884ae1171bfc72b2737974f5e8dec28479d
cd ..

source terryfy/travis_tools.sh

get_python_environment homebrew $PY_VERSION $(pwd)/_test_env
