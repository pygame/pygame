#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

# The latest cmake doesn't easily compile from source on centos 5
# So cmake is installed with pip
# This way we save compile time, reduce maintenance+scripting efforts and also 
# get the right binaries of the latest cmake version that will work on centos 5
# and above (the pip cmake package provides manylinux1 i686/x86_64 and 
# manylinux2014 i686/x86_64/aarch64 wheels)

# this file is only intended to work in our manylinux build system and not on
# MacOS, which has a modern enough cmake already

# any cpython version can be used here 
# (this must be updated when we drop 3.10 support after a few years)
PYTHON_VER=cp310-cp310
PYTHON_BIN=/opt/python/${PYTHON_VER}/bin

# this installs cmake in python bin dir, copy it to /usr/bin once installed
${PYTHON_BIN}/pip install cmake
cp ${PYTHON_BIN}/cmake /usr/bin
