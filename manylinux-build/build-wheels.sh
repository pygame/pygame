#!/bin/bash
set -e -x

SUPPORTED_PYTHONS="cp27-cp27mu cp34-cp34m cp35-cp35m"

export PORTMIDI_INC_PORTTIME=1

# Compile wheels
for PYVER in $SUPPORTED_PYTHONS; do
    rm -rf /io/Setup /io/build/
    PYBIN="/opt/python/${PYVER}/bin"
    ${PYBIN}/pip wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair $whl -w /io/manylinux-build/wheelhouse/
done

# Dummy options for headless testing
export SDL_AUDIODRIVER=disk
export SDL_VIDEODRIVER=dummy

# Install packages and test
for PYVER in $SUPPORTED_PYTHONS; do
    PYBIN="/opt/python/${PYVER}/bin"
    ${PYBIN}/pip install pygame --no-index -f /io/manylinux-build/wheelhouse
    (cd $HOME; ${PYBIN}/python -m pygame.tests --exclude opengl,music)
done
