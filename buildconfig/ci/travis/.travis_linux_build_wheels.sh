#!/bin/bash
set -e -x

# build the wheels.
cd buildconfig/manylinux-build
make pull
make wheels
cd ../..

mkdir -p dist/
cp buildconfig/manylinux-build/wheelhouse/*.whl dist/
