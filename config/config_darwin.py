import os, sys
from config import config_generic, config_unix

# Plistlib is always available on Mac OS X, but it is only
# available on Windows/Unix from 2.6
try: import plistlib
except: pass

def _get_framework_path(library):
    """
        Returns a tuple containing the path to the
        'Frameworks' folder of the OS and the path to
        the specific Framework for the given library.
    """
    for path in ('/', os.path.expanduser('~/'), '/System/'):
        path = os.path.join(path, 'Library/Frameworks/')
        framework = os.path.join(path,
                library + '.framework/Versions/Current/')

        framework_lib = os.path.join(framework, library)
        if os.path.isfile(framework_lib):
            return path, framework

    return (None, None)

def get_sys_libs(module):
    return []

def get_install_libs(cfg):
    return []

def sdl_get_version():
    """
        Returns the installed SDL version.
        In Mac OS, we first check for the version of the installed 
        SDL Framework bundle. If it's not available, we fallback to
        the version detection for all Unix systems.
    """
    path, framework = _get_framework_path('SDL')
    if framework:
        plist = plistlib.readPlist(os.path.join(framework, 'Resources', 'Info.plist'))
        return plist['CFBundleVersion'] + " (SDL.framework)"

    return config_unix.sdl_get_version()

class Dependency (config_unix.Dependency):
    """
        Mac OS X Library Dependency.

        Dependencies are handled exactly in the same manner as
        any other Unix system, except because we have to look in
        some additional system folders for libraries (the places
        where MacPorts and Fink install their stuff)
    """
    _searchdirs = ['/usr', '/usr/local', '/opt/local']
    _incdirs = ['include']
    _libdirs = ['lib']
    _libprefix = "lib"

    def _configure_frameworks(self):
        """
            Configuration callback which looks for installed SDL
            framework bundles.
        """
        path, framework = _get_framework_path(self.library_id)

        if not framework:
            return False

        self.incdirs.append(os.path.join(framework, 'Headers'))
        self.cflags.append('-Xlinker "-framework"')
        self.cflags.append('-Xlinker "%s"' % self.library_id)
        self.cflags.append('-Xlinker "-F%s"' % path)
        self.libs.remove(self.library_id)
        return True

    _configure_frameworks.priority = 4
