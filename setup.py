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
    
    # check to see if we are using mingw.
    import config
    # a separate method for finding dlls with mingw.
    if config.is_msys_mingw():

        print data_files
        # FIXME: hardcoding the dll paths for the moment.
        the_dlls = ["C:\\msys\\1.0\\local\\bin\\SDL.dll",
                    "C:\\msys\\1.0\\local\\bin\\SDL_image.dll",
                    "C:\\msys\\1.0\\local\\bin\\SDL_ttf.dll",
                    "C:\\msys\\1.0\\local\\bin\\SDL_mixer.dll",
                    "C:\\msys\\1.0\\local\\bin\\jpeg.dll",
                    "C:\\msys\\1.0\\local\\bin\\libpng13.dll",
                    "C:\\msys\\1.0\\local\\bin\\libtiff.dll",
                   ]

        the_dlls = ["C:\\MingW\\bin\\SDL.dll",
                    "C:\\MingW\\bin\\SDL_image.dll",
                    "C:\\MingW\\bin\\SDL_ttf.dll",
                    "C:\\MingW\\bin\\SDL_mixer.dll",
                    "C:\\MingW\\bin\\jpeg.dll",
                    "C:\\MingW\\bin\\libpng13.dll",
                    "C:\\MingW\\bin\\libtiff.dll",
                   ]


                   # no smpeg.
                    #"C:\\msys\\1.0\\local\\bin\\smpeg.dll"]

        data_files.extend(the_dlls)


        
        ext = "dll"
        for e in extensions:
            paths = []
            print e.library_dirs
            for d in e.library_dirs:
                 for l in e.libraries:
                        #name = tempcompiler.shared_lib_format%(l, ext)
                        name = "%s.%s" %(l, ext)
                        paths.append(os.path.join(d, name))
            #print paths
            for p in paths:
                if os.path.isfile(p) and p not in data_files:
                    data_files.append(p)


    else:

        tempcompiler = new_compiler()
        ext = tempcompiler.shared_lib_extension
        for e in extensions:
            paths = []
            print e.library_dirs
            for d in e.library_dirs:
                 for l in e.libraries:
                        name = tempcompiler.shared_lib_format%(l, ext)
                        paths.append(os.path.join(d, name))
            for p in paths:
                if os.path.isfile(p) and p not in data_files:
                    data_files.append(p)




#clean up the list of extensions
for e in extensions[:]:
    if e.name[:8] == 'COPYLIB_':
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
