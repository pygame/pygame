#!/usr/bin/env python

from distutils.core import setup, Extension
import os, glob


def get_c_files ():
    """Gets the list of files to use for building the module."""
    files = glob.glob (os.path.join ("src", "*.c"))
    return files

if __name__ == "__main__":

    extphysics = Extension ("physics", sources = get_c_files (),
                            include_dirs = [ "include" ],
                            extra_compile_args=["-g", "-W", "-Wall"])

    setupdata = {
        "name" : "physics",
        "version" : "0.0.1",
        "description" : "blabla",
        "license": "LGPL",
        "ext_modules" : [ extphysics ],
        }
    setup (**setupdata)
