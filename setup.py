#!/usr/bin/env python
#
# This is the distutils setup script for pygame.
# Full instructions are in "docs/fullinstall.txt"
#
# To configure, compile, install, just run this script.

METADATA = {
    "name":             "pygame",
    "version":          "1.0pre",
    "maintainer":       "Pete Shinners",
    "maintainer_email": "pygame@seul.org",
    "description":      "Python Game Development Package",
    "url":              "http://pygame.seul.org",
}



import os.path, glob, sys
import distutils.sysconfig 
from distutils.core import setup, Extension
from distutils.extension import read_setup_file
from distutils.ccompiler import new_compiler
from distutils.command.install_data import install_data


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
try:
    extensions = read_setup_file('Setup')
except:
    raise SystemExit, """Error with the "Setup" file,
perhaps make a clean copy from "Setup.in"."""


#extra files to install
data_path = os.path.join(distutils.sysconfig.get_python_lib(), 'pygame')
data_files = []


#add non .py files in lib directory
for f in glob.glob(os.path.join('lib', '*')):
    if not f.endswith('.py') and os.path.isfile(f):
        data_files.append(f)


#try to find DLLs and copy them too  (only on windows)
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


#clean up the list of extensions
for e in extensions[:]:
    if e.name.startswith('COPYLIB_'):
        extensions.remove(e) #don't compile the COPYLIBs, just clean them
    else:
        e.name = 'pygame.' + e.name #prepend package name on modules


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



#data installer with improved intelligence over distutils
#data files are copied into the project directory instead
#of willy-nilly
class smart_install_data(install_data):   
    def run(self):
        #need to change self.install_dir to the library dir

        print 'SMART_INSTALL_DATA'
        print '  old:', self.install_dir

        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        print '  new:', self.install_dir
        return install_data.run(self)

        
        #self.outfiles = []
        #install_cmd = self.get_finalized_command('install')
 
        #for d in self.data_files:
        #    d = self.check_data(d)
 
        #    install_dir = self.install_dir
            # alternative base dir given => overwrite install_dir
        #    if d.base_dir != None:
        #        install_dir = getattr(install_cmd,d.base_dir)
     
            # copy to an other directory 
        #    if d.copy_to != None:
        #        if not os.path.isabs(d.copy_to):
        #            # relatiev path to install_dir
        #            dir = os.path.join(install_dir, d.copy_to)
        #        elif install_cmd.root:
        #            # absolute path and alternative root set 
        #            dir = change_root(self.root,d.copy_to)
        #        else:
        #            # absolute path
        #            dir = d.copy_to                  
        #    else:
        #        # simply copy to install_dir
        #        dir = install_dir
        #        # warn if necceassary  
        #        self.warn("setup script did not provide a directory to copy files to "
        #                  " -- installing right in '%s'" % install_dir)
 
         #   dir=os.path.normpath(dir)
            # create path
         #   self.mkpath(dir) 
 




#finally, 
#call distutils with all needed info
PACKAGE_DATA = {
       "cmdclass":    {'install_data': smart_install_data},
       "packages":    ['pygame'],
       "package_dir": {'pygame': 'lib'},
       "headers":     headers,
       "ext_modules": extensions,
       "data_files":  [['pygame', data_files]],
}
PACKAGE_DATA.update(METADATA)
apply(setup, [], PACKAGE_DATA)


