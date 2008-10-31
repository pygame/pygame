import os, glob
from config import sdlconfig, pkgconfig, helpers, dll

def _hunt_libs (name, dirs):
    # Used by get_install_libs(). It resolves the dependency libraries
    # and returns them as dict.
    libs = {}
    x = dll.dependencies (name)
    for key in x.keys ():
        values = _get_libraries (key, dirs)
        libs.update (values)
    return libs

def _get_libraries (name, directories):
    # Gets the full qualified library path from directories.
    libs = {}
    dotest = dll.tester (name)
    for d in directories:
        try:
            files = os.listdir (d)
        except:
            pass
        else:
            for f in files:
                filename = os.path.join (d, f)
                if dotest (f) and os.path.isfile (filename):
                    # Found
                    libs[filename] = 1
    return libs

def find_incdir (name):
    # Gets the include directory for the specified header file.
    # TODO
    return None

def find_libdir (name):
    # Gets the library directory for the specified library file.
    # TODO
    return None

def sdl_get_version ():
    # Gets the SDL version.
    # TODO
    return None


def get_sys_libs (module):
    # Gets a list of system libraries to link the module against.
    if module == "sdl.scrap":
        return [ "-luser32", "-lgdi32" ]

def get_install_libs (cfg):
    # Gets the libraries to install for the target platform.
    # TODO
    return []
