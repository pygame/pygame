#!/bin/bash
set -e -x

SUPPORTED_PYTHONS="cp27-cp27mu cp34-cp34m cp35-cp35m cp36-cp36m cp37-cp37m cp38-cp38"

export PORTMIDI_INC_PORTTIME=1

# -msse4 is required by old gcc in centos, for the SSE4.2 used in image.c
# -g0 removes debugging symbols reducing file size greatly.
# -03 is full optimization on.
export CFLAGS="-msse4 -g0 -O3"

ls -la /io

# Compile wheels
for PYVER in $SUPPORTED_PYTHONS; do
    rm -rf /io/Setup /io/build/
    PYBIN="/opt/python/${PYVER}/bin"
    ${PYBIN}/pip wheel -vvv /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair $whl -w /io/buildconfig/manylinux-build/wheelhouse/
done

# Dummy options for headless testing
export SDL_AUDIODRIVER=disk
export SDL_VIDEODRIVER=dummy

# Install packages and test
for PYVER in $SUPPORTED_PYTHONS; do
    PYBIN="/opt/python/${PYVER}/bin"
    ${PYBIN}/pip install pygame --no-index -f /io/buildconfig/manylinux-build/wheelhouse
    (cd $HOME; ${PYBIN}/python -m pygame.tests --exclude opengl,music)
done
