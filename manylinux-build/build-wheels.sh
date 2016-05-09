#!/bin/bash
set -e -x

bash /io/manylinux-build/enable-repoforge.sh

# Install a system package required by our library
yum install -y SDL-devel libpng-devel libjpeg-devel libX11-devel freetype-devel \
                SDL_ttf-devel SDL_image-devel SDL_mixer-devel

SUPPORTED_PYTHONS="cp27-cp27mu cp34-cp34m cp35-cp35m"

# Compile wheels
for PYVER in $SUPPORTED_PYTHONS; do
    PYBIN="/opt/python/${PYVER}/bin"
    ${PYBIN}/pip wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair $whl -w /io/manylinux-build/wheelhouse/
done

# Install packages and test
for PYVER in $SUPPORTED_PYTHONS; do
    PYBIN="/opt/python/${PYVER}/bin"
    ${PYBIN}/pip install pygame --no-index -f /io/manylinux-build/wheelhouse
    (cd $HOME; ${PYBIN}/nosetests pymanylinuxdemo)
done
