#!/usr/bin/env python

import distutils.sysconfig
from distutils.core import setup
from distutils.command.install_data import install_data
import os, sys, glob, time, re
import modules, cfg
from config import helpers, msys, config_modules

VERSION = "2.0.0-alpha4"
DEBUG = True

# Minimum requirements.
PYTHON_MINIMUM = (2, 4)

# Data installer with improved intelligence over distutils.
# Data files are copied into the project directory instead of willy-nilly
class SmartInstallData(install_data):
    def run(self):
        # Need to change self.install_dir to the actual library dir.
        install_cmd = self.get_finalized_command ('install')
        self.install_dir = getattr (install_cmd, 'install_lib')
        return install_data.run (self)

def find_pkg_data (directory, excludedirs=[], excludefiles=[]):
    pkgdata = []
    nodirs = re.compile (r'(.svn)$')
    nofilesrev = re.compile (r'((~.*)|(cyp\..*)|(yp\..*))')
    for subd, directories, files in os.walk (directory):
        directories[:] = [d for d in directories if nodirs.match (d) is None \
                          and d not in excludedirs]
        files[:] = [f for f in files if nofilesrev.match(f[-1::-1]) is None \
                    and f not in excludefiles]
        subd = subd.replace (directory, "", 1)
        subd = subd.lstrip (os.path.sep)
        for f in files:
            pkgdata.append (os.path.join (subd, f))
    return pkgdata

def run_checks ():
    # Python version check.
    if helpers.getversion () < PYTHON_MINIMUM: # major, minor check
        raise Exception ("You should have at least Python >= %d.%d.x "
                         "installed." % PYTHON_MINIMUM)

    buildsystem = None
    builddefines = []
    if sys.platform == "win32":
        if msys.is_msys ():
            buildsystem = "msys"
            builddefines.append (("IS_MSYS", None))
        else:
            buildsystem = "win"
            builddefines.append (("IS_WIN32", None))
            builddefines.append (("WIN32", None))
    elif sys.platform == "darwin":
        buildsystem = "darwin"
        builddefines.append (("IS_DARWIN", None))
    else:
        buildsystem = "unix"
        builddefines.append (("IS_UNIX", None))

    if cfg.build['SDL']:
        sdlversion = config_modules.sdl_get_version (buildsystem)

    print ("\nThe following information will be used to build Pygame:")
    print ("\t System: %s" % buildsystem)
    print ("\t Python: %d.%d.%d" % helpers.getversion ())
    if cfg.build['SDL']:
        print ("\t SDL:    %s" % sdlversion)
    return buildsystem, builddefines

if __name__ == "__main__":

    buildsystem = None
    buildcflags = None
    try:
        buildsystem, builddefines = run_checks ()
    except:
        print (helpers.geterror ())
        print (helpers.gettraceback ())
        sys.exit (1)

    os.environ["CFLAGS"] = ""

    if "bdist_msi" in sys.argv:
        # hack the version name to a format msi doesn't have trouble with
        VERSION = VERSION.replace("-alpha", "a")
        VERSION = VERSION.replace("-rc", "r")

    if buildsystem in ("msys", "unix", "darwin") and DEBUG:
        os.environ["CFLAGS"] += " -W -Wall -Wimplicit-int " + \
                        "-Wimplicit-function-declaration " + \
                        "-Wimplicit -Wmain -Wreturn-type -Wunused " + \
                        "-Wswitch -Wcomment -Wtrigraphs -Wformat " + \
                        "-Wchar-subscripts -Wuninitialized -Wparentheses " +\
                        "-Wpointer-arith -Wcast-qual -Winline " + \
                        "-Wcast-align -Wconversion -Wstrict-prototypes " + \
                        "-Wmissing-prototypes -Wmissing-declarations " + \
                        "-Wnested-externs -Wshadow -Wredundant-decls -g -pg"

    # When building in Mac OS, we must make sure that all
    # modules are built as universal binaries.
    # FIXME: Find a better place for this?
    if buildsystem == "darwin":
        os.environ["CFLAGS"] += " -arch i386 -arch ppc"

    packages = [ "pygame2",
                 "pygame2.examples",
                 "pygame2.examples.freetype",
                 "pygame2.examples.sdl",
                 "pygame2.examples.sdlext",
                 "pygame2.examples.sdlgfx",
                 "pygame2.sprite",
                 "pygame2.threads",
                 "pygame2.test",
                 "pygame2.dll",
                 ]
    package_dir = { "pygame2" : "lib",
                    "pygame2.examples" : "examples",
                    "pygame2.sprite" : "lib/sprite",
                    "pygame2.threads" : "lib/threads",
                    "pygame2.test" : "test",
                    }
    package_data = {
        "pygame2.examples" : find_pkg_data ("examples"),
        "pygame2.test" : find_pkg_data ("test", ["c_api", "util"]),
        }

    dllfiles = [ os.path.join ("pygame2", "dll"),
                 config_modules.get_install_libs (buildsystem, cfg) ]
    ext_modules = modules.get_extensions (buildsystem)
    modules.update_packages (ext_modules, packages, package_dir, package_data)

    headerfiles = []
    print ("The following modules will be built:")
    for ext in ext_modules:
        ext.define_macros.extend (builddefines)
        headerfiles += ext.basemodule.installheaders
        print ("\t%s" % ext.name)

    # Allow the user to read what was printed
    time.sleep (2)

    # Create doc headers on demand.
    docincpath = os.path.join ("src", "doc")
    if not os.path.exists (docincpath):
        os.mkdir (docincpath)
    for ext in ext_modules:
        modules.create_docheader (ext.basemodule, docincpath)

    setupdata = {
        "cmdclass" : { "install_data" : SmartInstallData },
        "name" :  "pygame2",
        "version" : VERSION,
        "description" : "Python Game Development Library",
        "author" : "Pete Shinners, Rene Dudfield, Marcus von Appen, Lenard Lindstrom, Brian Fisher, others...",
        "author_email" : "pygame@seul.org",
        "license" : "LGPL",
        "url" : "http://pygame.org",
        "packages" : packages,
        "package_dir" : package_dir,
        "package_data" : package_data,
        "headers" : headerfiles,
        "ext_modules" : ext_modules,
        "data_files" : [ dllfiles ],
        }

    setup (**setupdata)
