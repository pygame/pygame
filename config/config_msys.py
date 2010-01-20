import os, glob
from config import msys, helpers, dll
from config import config_unix, config_generic, config_win, libconfig

try:
    msys_obj = msys.Msys (require_mingw=False)
except:
    msys_obj = None

def sdl_get_version ():
    return config_unix.sdl_get_version()

def add_sys_deps (module):
    config_win.add_sys_deps(module)

def update_sys_deps (deps):
    config_win.update_sys_deps(deps)

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
    _searchdirs = [ "/usr", "/usr/local", "/mingw" ]
    _incdirs = [ "include", "X11/include" ]
    _libdirs = [ "lib", "X11/lib" ]
    
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
    if cfg.build['FREETYPE']:
        libraries.update (_hunt_libs ("freetype", dirs))
    if cfg.build['PORTMIDI']:
        libraries.update (_hunt_libs ("portmidi", dirs))
    if cfg.build['OPENAL']:
        libraries.update (_hunt_libs ("openal", dirs))

    return [ k.replace ("/", os.sep) for k in libraries.keys() ]


class Dependency (config_unix.Dependency):
    _searchdirs = [ "/usr", "/usr/local", "/mingw" ]
    _incdirs = [ "include", "X11/include" ]
    _libdirs = [ "lib", "X11/lib" ]
    _libprefix = "lib"

    def _find_incdir (self, name):
        # Gets the include directory for the specified header file.
        def _fi_recurse(top):
            top = msys_obj.msys_to_windows (top)
            for (path, dirnames, filenames) in os.walk (top):
                if name in filenames:
                    return path
                for subfolder in dirnames:
                    _fi_recurse(os.path.join (path, subfolder))

        for d in self._searchdirs:
            for g in self._incdirs:
                p = _fi_recurse(os.path.join(d, g))
                if p:
                    return msys_obj.msys_to_windows (p)
        
    def _find_libdir (self, name):
        # Gets the library directory for the specified library file.
        for d in self._searchdirs:
            for g in self._libdirs:
                p = msys_obj.msys_to_windows (os.path.join (d, g))
                f = msys_obj.msys_to_windows (os.path.join (p, name))
                if list (filter (os.path.isfile, glob.glob (f + '*'))):
                    return p

    def _configure_libconfig(self):
        """
            Configuration callback using a generic CONFIG tool
        """
        lc = self.library_config_program
        found_header = False

        if not lc or not libconfig.has_libconfig(lc):
            return False

        incdirs = libconfig.get_incdirs(lc)
        for d in incdirs:
            for h in self.header_files:
                p = msys_obj.msys_to_windows (os.path.join(d, h))
                if os.path.isfile(p):
                    found_header = True
        if not found_header:
            return False

        self.incdirs += incdirs
        self.libdirs += libconfig.get_libdirs(lc)
        self.libs += libconfig.get_libs(lc)
        self.cflags += libconfig.get_cflags(lc)
        self.lflags += libconfig.get_lflags(lc)
        return True

    _configure_libconfig.priority = 2

    def configure(self, cfg):
        """
            Override the generic module configuration to make sure
            that all the found paths are converted from MSYS to full
            Windows path.
        """
        self.incdirs = [ msys_obj.msys_to_windows (d) for d in self.incdirs ]
        self.libdirs = [ msys_obj.msys_to_windows (d) for d in self.libdirs ]
        
        super(Dependency, self).configure(cfg)

        if self.configured:
            self.incdirs = [ msys_obj.msys_to_windows (d) for d in self.incdirs ]
            self.libdirs = [ msys_obj.msys_to_windows (d) for d in self.libdirs ]
