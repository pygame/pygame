#!/usr/bin/env python

from distutils.core import setup, Extension
import os, glob, sys


def get_c_files ():
    """Gets the list of files to use for building the module."""
    files = glob.glob (os.path.join ("src", "*.c"))
    return files

if __name__ == "__main__":

    warn_flags = ["-W", "-Wall", "-Wpointer-arith", "-Wcast-qual",
                  "-Winline", "-Wcast-align", "-Wconversion",
                  "-Wstrict-prototypes", "-Wmissing-prototypes",
                  "-Wmissing-declarations", "-Wnested-externs",
                  "-Wshadow", "-Wredundant-decls"
                  ]
    compile_args = [ "-std=c99", "-g"]
    compile_args += warn_flags

    extphysics = Extension ("physics", sources = get_c_files (),
                            include_dirs = [ "include" ],
                            define_macros = [("PHYSICS_INTERNAL", "1")],
                            extra_compile_args = compile_args)

    setupdata = {
        "name" : "physics",
        "version" : "0.0.1",
        "author" : "Zhang Fan",
        "url" : "http://www.pygame.org",
        "description" : "2D physics module",
        "license": "LGPL",
        "ext_modules" : [ extphysics ],
        "headers" : [ "include/pgphysics.h" ]
        }
    setup (**setupdata)
