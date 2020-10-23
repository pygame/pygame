#!/bin/bash
set -e -x

export SUPPORTED_PYTHONS="cp27-cp27mu cp35-cp35m cp36-cp36m cp37-cp37m cp38-cp38 cp39-cp39"

if [[ "$1" == "buildpypy" ]]; then
	export SUPPORTED_PYTHONS="pp27-pypy_73 pp36-pypy36_pp73 pp37-pypy37_pp73"
fi


export PORTMIDI_INC_PORTTIME=1

# To 'solve' this issue:
#   >>> process 338: D-Bus library appears to be incorrectly set up; failed to read
#   machine uuid: Failed to open "/var/lib/dbus/machine-id": No such file or directory
if [ ! -f /var/lib/dbus/machine-id ]; then
    dbus-uuidgen > /var/lib/dbus/machine-id
fi


# -msse4 is required by old gcc in centos, for the SSE4.2 used in image.c
# -g0 removes debugging symbols reducing file size greatly.
# -03 is full optimization on.
export CFLAGS="-g0 -O3"

ls -la /io

# Compile wheels
for PYVER in $SUPPORTED_PYTHONS; do
    rm -rf /io/Setup /io/build/
    PYBIN="/opt/python/${PYVER}/bin"
    PYTHON="/opt/python/${PYVER}/bin/python"
	if [ ! -f ${PYBIN}/python ]; then
	    PYTHON="/opt/python/${PYVER}/bin/pypy"
	fi

    ${PYTHON} -m pip wheel --global-option="build_ext" --global-option="-j4" -vvv /io/ -w wheelhouse/
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
    PYTHON="/opt/python/${PYVER}/bin/python"
	if [ ! -f ${PYBIN}/python ]; then
	    PYTHON="/opt/python/${PYVER}/bin/pypy"
	fi

    ${PYTHON} -m pip install pygame --no-index -f /io/buildconfig/manylinux-build/wheelhouse
    (cd $HOME; ${PYTHON} -m pygame.tests --exclude opengl,music,timing)
done
