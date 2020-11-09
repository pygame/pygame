#!/bin/bash
set -e -x

source build/conan/activate_run.sh

export CIBW_SKIP="pp27*"
# Divide the pythons between two travis jobs to avoid 50 minute timeout.
if [[ "${PYPYONLY}" == "yes" ]] && [[ -n "${TRAVIS_TAG}" ]]; then
  export CIBW_SKIP="pp27* cp*"
elif [[ -n "${TRAVIS_TAG}" ]]; then
  export CIBW_SKIP="pp*"
fi

$PYTHON_EXE -m cibuildwheel --output-dir dist
$PYTHON_EXE buildconfig/ci/travis/.travis_osx_rename_whl.py
