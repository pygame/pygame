#!/usr/bin/env python
# To use:
#       python setup.py install
#

morehelp = """
You can copy the file, "Setup.in" to "Setup" and configure
by hand. You can also run one of the configuration helper
scripts to do this for you: configwin.py or configunix.py"""


import os.path, glob, sys
import distutils.sysconfig 
from distutils.core import setup, Extension
from distutils.extension import read_setup_file
from distutils.ccompiler import new_compiler


#headers to install
headers = glob.glob(os.path.join('src', '*.h'))


#get compile info for all extensions
try:
    extensions = read_setup_file('Setup')
except IOError:
    raise SystemExit, 'Need a "Setup" file for compiling.' + morehelp


#extra files to install
data_path = os.path.join(distutils.sysconfig.get_python_lib(), 'pygame')
data_files = []

#add non .py files in lib directory
for f in glob.glob(os.path.join('lib', '*')):
    if not f.endswith('.py') and os.path.isfile(f):
        data_files.append(f)

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


#don't need to actually compile the COPYLIB modules, remove them
for e in extensions[:]:
    if e.name.startswith('COPYLIB_'):
        extensions.remove(e)


#we can detect the presence of python dependencies, remove any unfound
pythondeps = {'surfarray': ['Numeric']}
for e in extensions[:]:
    modules = pythondeps.get(e.name, [])
    if modules:
        try:
            for module in modules:
                x = __import__(module)
                del x
        except ImportError:
            print 'NOTE: Not compiling:', e.name, ' (module not found='+module+')'
            extensions.remove(e)

setup (name = "pygame",
       version = '0.9',
       maintainer = "Pete Shinners",
       maintainer_email = "pygame@seul.org",
       description = "Python Game Development module",
       url = "http://pygame.seul.org",
       packages = [''],
       package_dir = {'': 'lib'},
       extra_path = ('pygame/ignore', 'pygame'),
       headers = headers,
       ext_modules = extensions,
       data_files = [[data_path, data_files]]
     )  