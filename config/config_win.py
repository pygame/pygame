import os, glob
from config import sdlconfig, pkgconfig, helpers, dll

_searchdirs = [ "prebuilt", "..", "..\\.." ]
_incdirs = [ "", "include" ]
_libdirs = [ "", "VisualC\\SDL\\Release", "VisualC\\Release", "Release", "lib"]

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
            if len (list (filter (os.path.isfile, glob.glob (f + '*')))) > 0:
                return p

def sdl_get_version ():
    # Gets the SDL version.
    # TODO: Is there some way to detect the correct version?
    return "Unknown"

def get_sys_libs (module):
    # Gets a list of system libraries to link the module against.
    if module == "sdlext.scrap":
        return [ "user32", "gdi32" ]

def get_install_libs (cfg):
    # Gets the libraries to install for the target platform.
    libraries = {}
    values = {}
    dirs = []
    
    for d in _searchdirs:
        for g in _libdirs:
            dirs.append (os.path.join (d, g))
    
    if cfg.WITH_SDL:
        libraries.update (_hunt_libs ("SDL", dirs))
    if cfg.WITH_SDL_MIXER:
        libraries.update (_hunt_libs ("SDL_mixer", dirs))
    if cfg.WITH_SDL_IMAGE:
        libraries.update (_hunt_libs ("SDL_image", dirs))
    if cfg.WITH_SDL_TTF:
        libraries.update (_hunt_libs ("SDL_ttf", dirs))
    if cfg.WITH_SDL_GFX:
        libraries.update (_hunt_libs ("SDL_gfx", dirs))
    if cfg.WITH_PNG:
        libraries.update (_hunt_libs ("png", dirs))
    if cfg.WITH_JPEG:
        libraries.update (_hunt_libs ("jpeg", dirs))
    return libraries.keys ()
