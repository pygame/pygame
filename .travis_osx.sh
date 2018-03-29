#!/bin/bash

set -e

if [[ "$PY_VERSION" == "pypy2" ]] || [[ "$PY_VERSION" == "pypy3" ]]; then
	brew install $PY_VERSION
	export PYTHON_EXE=$PY_VERSION
	export PIP_CMD=$PYTHON_EXE -m pip
else

if [[ "$PY_VERSION" == "2" ]]; then
	brew install python@2
	export PYTHON_EXE=/usr/local/bin/python2
	export PIP_CMD=$PYTHON_EXE -m pip
fi
if [[ "$PY_VERSION" == "3" ]]; then
	brew install python
	export PYTHON_EXE=/usr/local/bin/python3
	export PIP_CMD=$PYTHON_EXE -m pip
fi

fi

