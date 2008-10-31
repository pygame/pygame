import os, glob
from config import sdlconfig, pkgconfig, helpers

_searchdirs = [ "/usr", "/usr/local" ]
_incdirs = [ "include", "X11/include" ]
_libdirs = [ "lib", "X11/lib" ]

def find_incdir (name):
    # Gets the include directory for the specified header file.
    for d in _searchdirs:
        for g in _incdirs:
            p = os.path.join (d, g)
            f = os.path.join (p, name)
            if os.path.isfile (f):
                return p

def find_libdir (name):
    # Gets the library directory for the specified library file.
    for d in _searchdirs:
        for g in _libdirs:
            p = os.path.join (d, g)
            f = os.path.join (p, name)
            if filter (os.path.isfile, glob.glob (f + '*')):
                return p

def sdl_get_version ():
    # Gets the SDL version.
    if pkgconfig.has_pkgconfig ():
        return pkgconfig.get_version ("sdl")[0]
    elif sdlconfig.has_sdlconfig ():
        return sdlconfig.get_version ()[0]
    return None

def get_sys_libs (module):
    # Gets a list of system libraries to link the module against.
    if module == "sdl.scrap":
        return [ "-lX11" ]

def get_install_libs (cfg):
    # Gets the libraries to install for the target platform.
    #
    # Assume plain shared libraries - do not add anything.
    return []
