import os, glob
from config import config_generic, sdlconfig, pkgconfig, helpers

def get_sys_libs (module):
    # Gets a list of system libraries to link the module against.
    if module == "sdlext.scrap":
        return [ "X11" ]

def sdl_get_version():
    """
        Returns the installed SDL library version.
        By default, we obtain the version using either
        'pkgconfig' or the 'sdl-config' tool.
    """
    if pkgconfig.has_pkgconfig ():
        return pkgconfig.get_version ("sdl")[0]
    elif sdlconfig.has_sdlconfig ():
        return sdlconfig.get_version ()[0]

    # TODO: SDL may be installed manually (i.e. compiled from
    # source). any way to find the version?
    return None

def get_install_libs(cfg):
    return []


class Dependency (config_generic.Dependency):
    _searchdirs = [ "/usr", "/usr/local" ]
    _incdirs = [ "include", "X11/include" ]
    _libdirs = [ "lib", "X11/lib" ]

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
                if filter (os.path.isfile, glob.glob (f + '*')):
                    return p

    def _configure_pkgconfig(self):
        """
            Configuration callback using the 'pkgconfig' tool
        """
        if (not pkgconfig.has_pkgconfig() or 
            not pkgconfig.exists(self.library_name)):
            return False

        self.incdirs += pkgconfig.get_incdirs(self.library_name)
        self.libdirs += pkgconfig.get_libdirs(self.library_name)
        self.libs += pkgconfig.get_libs(self.library_name)
        self.cflags += pkgconfig.get_cflags(self.library_name)
        self.lflags += pkgconfig.get_lflags(self.library_name)
        return True

    _configure_pkgconfig.priority = 1

class DependencySDL (config_generic.DependencySDL, Dependency):
    # look in the SDL subdir for headers
    _incdirs = [ "include", "X11/include", "include/SDL", "X11/include/SDL" ]

    def _configure_pkgconfig(self):
        """
            Configuration callback using the 'pkgconfig' tool.
            Note that all the SDL-based libraries don't show up
            as such under pkgconfig, hence we need to always look
            for SDL.
        """
        if (not pkgconfig.has_pkgconfig() or 
            not pkgconfig.exists('sdl')):
            return False

        if not self._find_incdir(self.header_file):
            return False

        self.incdirs += pkgconfig.get_incdirs('sdl')
        self.libdirs += pkgconfig.get_libdirs('sdl')
        self.libs += pkgconfig.get_libs('sdl')
        self.cflags += pkgconfig.get_cflags('sdl')
        self.lflags += pkgconfig.get_lflags('sdl')
        return True

    _configure_pkgconfig.priority = 1

    def _configure_sdlconfig(self):
        """
            Configuration callback using the 'sdl-config' tool
        """
        if not sdlconfig.has_sdlconfig():
            return False

        # SDL-config returns valid values for all the sdl-based
        # libraries (sdl_ttf, sdl_mixer, etc) even if the library
        # is not installed. Make sure at least that its header
        # exists!
        if not self._find_incdir(self.header_file):
            return False

        self.incdirs += sdlconfig.get_incdirs ()
        self.libdirs += sdlconfig.get_libdirs ()
        self.libs += sdlconfig.get_libs ()
        self.cflags += sdlconfig.get_cflags ()
        self.lflags += sdlconfig.get_lflags ()
        return True

    _configure_sdlconfig.priority = 2

