#!/bin/bash
set -e -x

cd /pypy_build/

PYPY27="pypy2.7-v7.3.5-linux64"
PYPY37="pypy3.7-v7.3.5-linux64"

curl -sL https://downloads.python.org/pypy/${PYPY27}.tar.bz2 > ${PYPY27}.tar.bz2
curl -sL https://downloads.python.org/pypy/${PYPY37}.tar.bz2 > ${PYPY37}.tar.bz2
sha512sum -c pypy64.sha512

mkdir -p /opt/python/pp27-pypy_75/
mkdir -p /opt/python/pp37-pypy37_pp75/
tar xvf ${PYPY27}.tar.bz2 -C /opt/python/pp27-pypy_75/ --strip 1
tar xvf ${PYPY37}.tar.bz2 -C /opt/python/pp37-pypy37_pp75/ --strip 1

/opt/python/pp27-pypy_75/bin/pypy -m ensurepip
/opt/python/pp37-pypy37_pp75/bin/pypy -m ensurepip

/opt/python/pp27-pypy_75/bin/pypy -m pip install wheel
/opt/python/pp37-pypy37_pp75/bin/pypy -m pip install wheel

cd ..