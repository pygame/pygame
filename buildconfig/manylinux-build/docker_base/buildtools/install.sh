#!/bin/bash

# This file installs tools (cmake and meson) needed to build dependencies
# TODO: This script should also be run in mac buildconfig later on

set -e -x

cd $(dirname `readlink -f "$0"`)

# The latest cmake doesn't easily compile from source on centos 5
# So cmake is installed with pip
# This way we save compile time, reduce maintenance+scripting efforts and also 
# get the right binaries of the latest cmake version that will work on centos 5
# and above (the pip cmake package provides manylinux1 i686/x86_64 and 
# manylinux2014 i686/x86_64/aarch64 wheels)

# pin versions for stability (remember to keep updated)
python3 -m pip install --user cmake==3.24.1.1 meson==0.63.2 ninja==1.10.2.3

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cp /root/.local/bin/* /usr/bin
fi
