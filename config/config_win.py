import os, glob
from config import sdlconfig, pkgconfig, helpers, dll
from config import config_generic

def sdl_get_version ():
    # Gets the SDL version.
    # TODO: Is there some way to detect the correct version?
    return "Unknown"

def get_sys_libs (module):
    # Gets a list of system libraries to link the module against.
    if module == "sdlext.scrap":
        return [ "user32", "gdi32" ]

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

def get_install_libs (cfg):
    # Gets the libraries to install for the target platform.
    _libdirs = [ "", "VisualC\\SDL\\Release", "VisualC\\Release", "Release",
                 "lib"]
    _searchdirs = [ "prebuilt", "..", "..\\.." ]

    libraries = {}
    values = {}
    dirs = []
    
    for d in _searchdirs:
        for g in _libdirs:
            dirs.append (os.path.join (d, g))
    
    if cfg.build['SDL']:
        libraries.update (_hunt_libs ("SDL", dirs))
    if cfg.build['SDL_MIXER']:
        libraries.update (_hunt_libs ("SDL_mixer", dirs))
    if cfg.build['SDL_IMAGE']:
        libraries.update (_hunt_libs ("SDL_image", dirs))
    if cfg.build['SDL_TTF']:
        libraries.update (_hunt_libs ("SDL_ttf", dirs))
    if cfg.build['SDL_GFX']:
        libraries.update (_hunt_libs ("SDL_gfx", dirs))
    if cfg.build['PNG']:
        libraries.update (_hunt_libs ("png", dirs))
    if cfg.build['JPEG']:
        libraries.update (_hunt_libs ("jpeg", dirs))

    return libraries.keys ()


class Dependency (config_generic.Dependency):
    _searchdirs = [ "prebuilt", "..", "..\\.." ]
    _incdirs = [ "", "include" ]
    _libdirs = [ "", "VisualC\\SDL\\Release", "VisualC\\Release", "Release",
                 "lib"]
    
    def __init__(self, header_file, library_link_id):
        super (Dependency, self).__init__ (header_file, library_link_id)
        self.library_name = library_link_id
        
    def _find_incdir(self, name):
        # Gets the include directory for the specified header file.
        for d in self._searchdirs:
            for g in self._incdirs:
                p = os.path.join (d, g)
                f = os.path.join (p, name)
                if os.path.isfile (f):
                    return p

    def _find_libdir(self, name):
        # Gets the library directory for the specified library file.
        for d in self._searchdirs:
            for g in self._libdirs:
                p = os.path.join (d, g)
                f = os.path.join (p, name)
                if len (list (filter (os.path.isfile, glob.glob (f + '*')))) > 0:
                    return p

class DependencySDL(config_generic.DependencySDL, Dependency):
    def __init__(self, header_file, library_link_id):
        super (DependencySDL, self).__init__ (header_file, library_link_id)
        self.library_name = library_link_id
    
    def _configure_guess(self):
        if super(DependencySDL, self)._configure_guess():
            self.libs.append('SDLmain')
            ldir = self._find_libdir ('SDL')
            if ldir is not None:
                self.libdirs.append (ldir)
            return True

        return False

    # Under windows, always guess the library position
    _configure_guess.priority = 5

