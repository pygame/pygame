#!/bin/bash
set -e -x

$PYTHON_EXE setup.py sdist
$PYTHON_EXE -m pip install dist/*.tar.gz
$PYTHON_EXE -m pygame.tests.__main__ -v --exclude opengl,timing --time_out 300
