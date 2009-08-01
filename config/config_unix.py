import os, glob, sys
from config import config_generic, libconfig, pkgconfig, helpers

def update_sys_deps (deps):
    deps["x11"] = Dependency (['Xutil.h'], 'X11', pkgconfig_name='x11')

def add_sys_deps (module):
    if module.name == "sdlext.scrap":
        module.depends.append ("x11")

def sdl_get_version():
    """
        Returns the installed SDL library version.
        By default, we obtain the version using either
        'pkgconfig' or the 'sdl-config' tool.
    """
    if pkgconfig.has_pkgconfig ():
        return pkgconfig.get_version ("sdl")[0]
    elif libconfig.has_libconfig("sdl-config"):
        return libconfig.get_version("sdl-config")[0]

    # TODO: SDL may be installed manually (i.e. compiled from
    # source). any way to find the version?
    return None

def get_install_libs(cfg):
    return []

class Dependency (config_generic.Dependency):
    _searchdirs = [ "/usr", "/usr/local", sys.prefix ]
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

    _configure_pkgconfig.priority = 2

    def _configure_libconfig(self):
        """
            Configuration callback using a generic CONFIG tool
        """
        lc = self.library_config_program
        if not lc or not libconfig.has_libconfig(lc):
            return False

        self.incdirs += libconfig.get_incdirs(lc)
        self.libdirs += libconfig.get_libdirs(lc)
        self.libs += libconfig.get_libs(lc)
        self.cflags += libconfig.get_cflags(lc)
        self.lflags += libconfig.get_lflags(lc)
        return True

    _configure_libconfig.priority = 1

