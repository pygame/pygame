#!/usr/bin/env python

import distutils.sysconfig
from distutils.core import setup
from distutils.command.install_data import install_data
import os, sys, glob, time
import modules, cfg
from config import *

VERSION = "2.0.0"

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

def run_checks ():
    # Python version check.
    if helpers.getversion () < PYTHON_MINIMUM: # major, minor check
        raise Exception ("You should have at least Python >= %d.%d.x "
                         "installed." % PYTHON_MINIMUM)

    buildsystem = None
    if sys.platform == "win32":
        if msys.is_msys ():
            buildsystem = "msys"
        else:
            buildsystem = "win"
    else:
        buildsystem = "unix"

    if cfg.WITH_SDL:
        sdlversion = config_modules.sdl_get_version (buildsystem)

    print ("\nThe following information will be used to build Pygame:")
    print ("\t System: %s" % buildsystem)
    print ("\t Python: %d.%d.%d" % helpers.getversion ())
    if cfg.WITH_SDL:
        print ("\t SDL:    %s" % sdlversion)
    return buildsystem

if __name__ == "__main__":

    buildsystem = None
    try:
        buildsystem = run_checks ()
    except:
        print (helpers.geterror ())
        sys.exit (1)

    if buildsystem in ("msys", "unix"):
        os.environ["CFLAGS"] = "-W -Wall -Wimplicit-int " + \
                        "-Wimplicit-function-declaration " + \
                        "-Wimplicit -Wmain -Wreturn-type -Wunused " + \
                        "-Wswitch -Wcomment -Wtrigraphs -Wformat " + \
                        "-Wchar-subscripts -Wuninitialized -Wparentheses " +\
                        "-Wpointer-arith -Wcast-qual -Winline " + \
                        "-Wcast-align  -Wconversion -Wstrict-prototypes " + \
                        "-Wmissing-prototypes -Wmissing-declarations " + \
                        "-Wnested-externs -Wshadow -Wredundant-decls -g -pg"

    packages = [ "pygame2",
                 "pygame2.sprite",
                 "pygame2.threads",
                 "pygame2.test",
                 "pygame2.dll",
                 ]
    package_dir = { "pygame2" : "lib",
                    "pygame2.sprite" : "lib/sprite",
                    "pygame2.threads" : "lib/threads",
                    "pygame2.test" : "test",
                    }
    package_data = {}
    modules.update_packages (cfg, packages, package_dir, package_data)

    dllfiles = [ os.path.join ("pygame2", "dll"),
                 config_modules.get_install_libs (buildsystem, cfg) ]
    ext_modules = modules.get_extensions (buildsystem)
    headerfiles = []
    print ("The following modules will be built:")
    for ext in ext_modules:
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
