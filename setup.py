#!/usr/bin/env python
# To use:
#       python setup.py install
#
import os.path, glob, sys
from distutils.core import setup, Extension
from distutils.extension import read_setup_file
from distutils.ccompiler import new_compiler
#from distutils.sysconfig import get_python_inc

#headers to install
headers = glob.glob(os.path.join('src', '*.h'))


#get compile info for all extensions
try:
    extensions = read_setup_file('Setup')
except IOError:
    raise SystemExit, \
"""Need a valid "Setup" file for compiling.
Make of copy of "Setup.in" end edit,
or run "confighelp.py" to create a new one."""


#extra files to install
data_files = []



#try to find libaries and copy them too
#(great for windows, bad for linux)
if sys.platform == 'win32':
    tempcompiler = new_compiler()
    for e in extensions:
        paths = [os.path.join(d, \
                 tempcompiler.shared_lib_format%(l, \
                     tempcompiler.shared_lib_extension)) \
                 for d in e.library_dirs for l in e.libraries]
        for p in paths:
            if os.path.isfile(p) and p not in data_files:
                data_files.append(p)


#don't need to actually compile the COPYLIB modules
for e in extensions[:]:
    if e.name[:8] == 'COPYLIB_':
        extensions.remove(e)




setup (name = "pygame",
       version = '0.2',
       maintainer = "Pete Shinners",
       maintainer_email = "shredwheat@mediaone.net",
       description = "Python Game Development module",
       url = "http://pygame.seul.org",

       packages = [''],
       package_dir = {'': 'lib'},
       extra_path = ('pygame/ignore', 'pygame'),

       headers = headers,
       ext_modules = extensions,
       data_files = [['pygame', data_files]]
       )

