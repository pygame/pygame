import os, glob
from config import sdlconfig, pkgconfig, msys, helpers, dll
from config import config_unix, config_generic

try:
    msys_obj = msys.Msys (require_mingw=False)
except:
    msys_obj = None

def sdl_get_version ():
    return config_unix.sdl_get_version()

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
    libraries = {}
    values = {}
    
    dirs = []
    for d in _searchdirs:
        for g in _libdirs:
            dirs.append (msys_obj.msys_to_windows (os.path.join (d, g)))
        dirs.append (msys_obj.msys_to_windows (os.path.join (d, "bin")))
    dirs += [ msys_obj.msys_to_windows ("/mingw/" + d) for d in _libdirs]
    dirs += [ msys_obj.msys_to_windows ("/mingw/bin") ]
    
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

    return [ k.replace ("/", os.sep) for k in libraries.keys() ]


class Dependency (config_unix.Dependency):
    _searchdirs = [ "/usr", "/usr/local", "/mingw" ]
    _incdirs = [ "include", "X11/include" ]
    _libdirs = [ "lib", "X11/lib" ]

    def _find_incdir (self, name):
        # Gets the include directory for the specified header file.
        for d in self._searchdirs:
            for g in self._incdirs:
                p = msys_obj.msys_to_windows (os.path.join (d, g))
                f = os.path.join (p, name)
                if os.path.isfile (f):
                    return p

    def _find_libdir (self, name):
        # Gets the library directory for the specified library file.
        for d in self._searchdirs:
            for g in self._libdirs:
                p = msys_obj.msys_to_windows (os.path.join (d, g))
                f = os.path.join (p, name)
                if filter (os.path.isfile, glob.glob (f + '*')):
                    return p

    def configure(self, cfg):
        """
            Override the generic module configuration to make sure
            that all the found paths are converted from MSYS to full
            Windows path.
        """
        super(Dependency, self).configure(cfg)

        if self.configured:
            self.incdirs = [ msys_obj.msys_to_windows (d) for d in self.incdirs ]
            self.libdirs = [ msys_obj.msys_to_windows (d) for d in self.libdirs ]

class DependencySDL(config_unix.DependencySDL, Dependency):
    def _configure_guess(self):
        if super(Dependency, self)._configure_guess():
            self.libs.append('SDLmain')
            self.libdirs.append(self._find_libdir('SDL'))
            return True

        return False

    # Always attempt to configure first
    _configure_guess.priority = 5

