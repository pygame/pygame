#!/bin/bash
set -e -x

cd $(dirname `readlink -f "$0"`)

# pin for build stability, remember to keep updated
python3 -m pip install mako==1.2.3
