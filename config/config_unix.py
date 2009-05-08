import os, glob
from config import config_generic, libconfig, pkgconfig, helpers

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
    elif libconfig.has_sdlconfig("sdl"):
        return libconfig.get_version("sdl")[0]

    # TODO: SDL may be installed manually (i.e. compiled from
    # source). any way to find the version?
    return None

def get_install_libs(cfg):
    return []


class Dependency (config_generic.Dependency):
    _searchdirs = [ "/usr", "/usr/local" ]
    _incdirs = [ "include", "X11/include" ]
    _libdirs = [ "lib", "X11/lib" ]
    _libprefix = "lib"

    def _configure_pkgconfig(self):
        """
            Configuration callback using the 'pkgconfig' tool
        """

        pkg = self.pkgconfig_name

        if (not pkg or
            not pkgconfig.has_pkgconfig() or 
            not pkgconfig.exists(pkg)):
            return False

        self.incdirs += pkgconfig.get_incdirs(pkg)
        self.libdirs += pkgconfig.get_libdirs(pkg)
        self.libs += pkgconfig.get_libs(pkg)
        self.cflags += pkgconfig.get_cflags(pkg)
        self.lflags += pkgconfig.get_lflags(pkg)
        return True

    _configure_pkgconfig.priority = 1

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
            if os.path.isfile(os.path.join(d, self.header_file)):
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

