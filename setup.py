#!/usr/bin/env python
#
# This is the distutils setup script for pygame.
# Full instructions are in "install.txt" or "install.html"
#
# To configure, compile, install, just run this script.

DESCRIPTION = """Pygame is a Python wrapper module for the
SDL multimedia library. It contains python functions and classes
that will allow you to use SDL's support for playing cdroms,
audio and video output, and keyboard, mouse and joystick input."""

EXTRAS = {}

METADATA = {
    "name":             "pygame",
    "version":          "1.8.0rc3",
    "license":          "LGPL",
    "url":              "http://www.pygame.org",
    "author":           "Pete Shinners",
    "author_email":     "pygame@seul.org",
    "description":      "Python Game Development",
    "long_description": DESCRIPTION,
}

import sys
if not hasattr(sys, 'version_info') or sys.version_info < (2,2):
    raise SystemExit, "Pygame requires Python version 2.2 or above."

#get us to the correct directory
import os, sys
path = os.path.split(os.path.abspath(sys.argv[0]))[0]
os.chdir(path)
#os.environ["CFLAGS"] = "-W -Wall -Wpointer-arith -Wcast-qual -Winline " + \
#                       "-Wcast-align -Wconversion -Wstrict-prototypes " + \
#                       "-Wmissing-prototypes -Wmissing-declarations " + \
#                       "-Wnested-externs -Wshadow -Wredundant-decls"
if "-warnings" in sys.argv:
    os.environ["CFLAGS"] = "-W -Wimplicit-int " + \
                       "-Wimplicit-function-declaration " + \
                       "-Wimplicit -Wmain -Wreturn-type -Wunused -Wswitch " + \
                       "-Wcomment -Wtrigraphs -Wformat -Wchar-subscripts " + \
                       "-Wuninitialized -Wparentheses " +\
                       "-Wpointer-arith -Wcast-qual -Winline -Wcast-align " + \
                       "-Wconversion -Wstrict-prototypes " + \
                       "-Wmissing-prototypes -Wmissing-declarations " + \
                       "-Wnested-externs -Wshadow -Wredundant-decls"
    sys.argv.remove ("-warnings")

import os.path, glob
import distutils.sysconfig
from distutils.core import setup, Extension
from distutils.extension import read_setup_file
from distutils.ccompiler import new_compiler
from distutils.command.install_data import install_data

try:
    import bdist_mpkg_support
    from setuptools import setup, Extension
except ImportError:
    pass
else:
    EXTRAS.update({
        'options': bdist_mpkg_support.options,
        'setup_requires': ['bdist_mpkg>=0.4.2'],
    })

import config
# a separate method for finding dlls with mingw.
if config.is_msys_mingw():

	# fix up the paths for msys compiling.
	import distutils_mods
	distutils.cygwinccompiler.Mingw32 = distutils_mods.mingcomp



#headers to install
headers = glob.glob(os.path.join('src', '*.h'))
headers.remove(os.path.join('src', 'numeric_arrayobject.h'))

#sanity check for any arguments
if len(sys.argv) == 1:
    reply = raw_input('\nNo Arguments Given, Perform Default Install? [Y/n]')
    if not reply or reply[0].lower() != 'n':
        sys.argv.append('install')


#make sure there is a Setup file
if not os.path.isfile('Setup'):
    print '\n\nWARNING, No "Setup" File Exists, Running "config.py"'
    import config
    config.main()
    print '\nContinuing With "setup.py"'



#get compile info for all extensions
try: extensions = read_setup_file('Setup')
except: raise SystemExit, """Error with the "Setup" file,
perhaps make a clean copy from "Setup.in"."""


#extra files to install
data_path = os.path.join(distutils.sysconfig.get_python_lib(), 'pygame')
data_files = []


#add non .py files in lib directory
for f in glob.glob(os.path.join('lib', '*')):
    if not f[-3:] =='.py' and os.path.isfile(f):
        data_files.append(f)


#try to find DLLs and copy them too  (only on windows)
if sys.platform == 'win32':

    #add dependency DLLs to the project
    import dll

    the_dlls = {}
    required_dlls = {}
    for e in extensions:
        if e.name.startswith('COPYLIB_'):
            the_dlls[e.libraries[0]] = e.library_dirs[0]
        else:
            required_dlls.update(dll.dependencies(e.libraries))
    for lib in required_dlls:
        #next DLL; a distutils bug requires the paths to have Windows separators
        f = the_dlls[lib].replace('/', os.sep)
        if f == '_':
            print "WARNING, DLL for %s library not found." % lib
        else:
            data_files.append(f)



#clean up the list of extensions
for e in extensions[:]:
    if e.name.startswith('COPYLIB_'):
        extensions.remove(e) #don't compile the COPYLIBs, just clean them
    else:
        e.name = 'pygame.' + e.name #prepend package name on modules


#data installer with improved intelligence over distutils
#data files are copied into the project directory instead
#of willy-nilly
class smart_install_data(install_data):
    def run(self):
        #need to change self.install_dir to the actual library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)

#finally,
#call distutils with all needed info
PACKAGEDATA = {
       "cmdclass":    {'install_data': smart_install_data},
       "packages":    ['pygame', 'pygame.gp2x'],
       "package_dir": {'pygame': 'lib',
                       'pygame.gp2x': 'lib/gp2x'},
       "headers":     headers,
       "ext_modules": extensions,
       "data_files":  [['pygame', data_files]],
}
PACKAGEDATA.update(METADATA)
PACKAGEDATA.update(EXTRAS)
setup(**PACKAGEDATA)
