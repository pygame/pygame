#!/bin/bash
# $Id:$

pythons="python2.3 python2.4 python2.5"

for python in $pythons; do
    $python setup_sdl.py bdist_egg
    $python setup_pygame.py bdist_egg
done
