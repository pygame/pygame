#!/bin/bash

# This is used to reduce the size of our manylinux wheels. This works by
# stripping unneeded symbols and elf sections (including debug symbols and the
# like).

# This is a pretty scary looking command, let's break it down part-wise to
# understand it

# > find /usr/local/lib
# searches everything recursively under /usr/local/lib (including the top dir)

# > \( -name "*.so*" -o -name "*.a" \)
# matches names having a .so[suffix] OR .a extension

# > -type f -xtype f
# matches only files and excludes symbolic links

# > -print0
# This makes find null-terminate every individual entry it finds. This properly
# handles cases where file names have newlines or whitespaces

# > xargs -0 -t
# xargs is responsible for running a command on the output of find. The -0
# option tells it that entries are null terminated, and -t prints information
# to stderr for debugging purposes

# > strip --strip-unneeded
# This is the actual command being run on all so files: this strips unneeded
# info
find /usr/local/lib \( -name "*.so*" -o -name "*.a" \) -type f -xtype f -print0 | \
    xargs -0 -t strip --strip-unneeded
