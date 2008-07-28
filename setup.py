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
    "version":          "1.8.1release",
    "license":          "LGPL",
    "url":              "http://www.pygame.org",
    "author":           "Pete Shinners, Rene Dudfield, Marcus von Appen, Bob Pendleton, others...",
    "author_email":     "pygame@seul.org",
    "description":      "Python Game Development",
    "long_description": DESCRIPTION,
}

import sys

if "bdist_msi" in sys.argv:
    # hack the version name to a format msi doesn't have trouble with
    METADATA["version"] = METADATA["version"].replace("pre", "a0")
    METADATA["version"] = METADATA["version"].replace("rc", "b0")
    METADATA["version"] = METADATA["version"].replace("release", "c0s")
    

if not hasattr(sys, 'version_info') or sys.version_info < (2,3):
    raise SystemExit, "Pygame requires Python version 2.3 or above."

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

import os.path, glob, stat
import distutils.sysconfig
from distutils.core import setup, Extension
from distutils.extension import read_setup_file
from distutils.command.install_data import install_data

# NOTE: the pyobjc, and bdist_mpkg_support is for darwin.
try:
    import bdist_mpkg_support
    from setuptools import setup, Extension
except ImportError:
    pass
else:
    EXTRAS.update({
        'options': bdist_mpkg_support.options,
        'setup_requires': ['bdist_mpkg>=0.4.2'],
        #'install_requires': ['pyobjc'],
        #'dependency_links': ['http://rene.f0o.com/~rene/stuff/macosx/']
    })

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


try:
    s_mtime = os.stat("Setup")[stat.ST_MTIME]
    sin_mtime = os.stat("Setup.in")[stat.ST_MTIME]
    if sin_mtime > s_mtime:
        print '\n\nWARNING, "Setup.in" newer than "Setup", you might need to modify Setup."'
except:
    pass

#get compile info for all extensions
try: extensions = read_setup_file('Setup')
except: raise SystemExit, """Error with the "Setup" file,
perhaps make a clean copy from "Setup.in"."""


#extra files to install
data_path = os.path.join(distutils.sysconfig.get_python_lib(), 'pygame')
data_files = []


#add non .py files in lib directory
for f in glob.glob(os.path.join('lib', '*')):
    if not f[-3:] == '.py' and not f[-4:] == '.doc' and os.path.isfile(f):
        data_files.append(f)

# Required. This will be filled if doing a Windows build.
cmdclass = {}

#try to find DLLs and copy them too  (only on windows)
if sys.platform == 'win32':

    from distutils.command.build_ext import build_ext
    # mingw32distutils is optional. But we need the mingw32 compiler(s).
    try:
        # Allow the choice between Win32 GUI and console DLLs.
        import mingw32distutils
    except ImportError:
        mingw32_compilers = ['ming32']
    else:
        mingw32_compilers = mingw32distutils.compilers
    if sys.version_info < (2, 4):
        try:
            # !!! This part looks very outdated. distutils_mods does not
            # even show up on a Google search. It can probably go at some
            # point. It is not needed for Python 2.4 and higher.
            import config
            # a separate method for finding dlls with mingw.
            if config.is_msys_mingw():

                # fix up the paths for msys compiling.
                import distutils_mods
                distutils.cygwinccompiler.Mingw32 = distutils_mods.mingcomp
        except ImportError:
            pass
        
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

    class WinBuildExt(build_ext):
        """This build_ext sets necessary environment variables for MinGW"""

        # __sdl_lib_dir is possible location of msvcrt replacement import
        # libraries, if they exist. Pygame module base only links to SDL so
        # should have the SDL library directory as its only -L option.
        for e in extensions:
            if e.name == 'base':
                __sdl_lib_dir = e.library_dirs[0].replace('/', os.sep)
                break
        
        def run(self):
            """Extended to set MINGW_ROOT_DIRECTORY, PATH and LIBRARY_PATH"""
            
            if self.compiler in mingw32_compilers:
                # Add MinGW environment variables.
                if 'MINGW_ROOT_DIRECTORY' not in os.environ:
                    # Use MinGW setup conifiguration file if present.
                    import mingwcfg
                    try:
                        mingw_root = mingwcfg.read()
                    except IOError:
                        raise RuntimeError(
                            "mingw32: required environment variable"
                            " MINGW_ROOT_DIRECTORY not set")
                    os.environ['MINGW_ROOT_DIRECTORY'] = mingw_root
                    path = os.environ['PATH']
                    os.environ['PATH'] = ';'.join([os.path.join(mingw_root, 'bin'),
                                                   path])
                if sys.version_info >= (2, 6):
                    # The Visual Studio 2008 C library is msvcr90.dll.
                    c_runtime_path = os.path.join(self.__sdl_lib_dir, 'msvcr90')
                elif sys.version_info >= (2, 4):
                    # The Visual Studio 2003 C library is msvcr71.dll.
                    c_runtime_path = os.path.join(self.__sdl_lib_dir, 'msvcr71')
                else:
                    # The Visual Studio 6.0 C library is msvcrt.dll,
                    # the MinGW default.
                    c_runtime_path = ''
                if c_runtime_path and os.path.isdir(c_runtime_path):
                    # Override the default msvcrt.dll linkage.
                    os.environ['LIBRARY_PATH'] = c_runtime_path
                elif not (c_runtime_path or
                          glob.glob(os.path.join(self.__sdl_lib_dir,
                                                 'msvcr*'))):
                    pass
                else:
                    raise RuntimeError("The dependencies are linked to"
                                       " the wrong C runtime for"
                                       " Python %i.%i" %
                                       sys.version_info[:2])
            build_ext.run(self)
    cmdclass['build_ext'] = WinBuildExt



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

cmdclass['install_data'] = smart_install_data

            
if "bdist_msi" in sys.argv:
    # if you are making an msi, we want it to overwrite files
    from distutils.command import bdist_msi
    import msilib

    class bdist_msi_overwrite_on_install(bdist_msi.bdist_msi):
        def run(self):
            bdist_msi.bdist_msi.run(self)

            # Remove obsolete files.
            comp = "pygame1"  # Pygame component
            prop = comp   # Directory property
            records = [("surfarray.pyd", comp,
                        "SURFAR~1.PYD|surfarray.pyd", prop, 1),
                       ("sndarray.pyd", comp,
                        "SNDARRAY.PYD|sndarray.pyd", prop, 1),
                       ("color.py", comp, "COLOR.PY|color.py", prop, 1),
                       ("color.pyc", comp, "COLOR.PYC|color.pyc", prop, 1),
                       ("color.pyo", comp, "COLOR.PYO|color.pyo", prop, 1)]
            msilib.add_data(self.db, "RemoveFile", records)

            # Overwrite outdated files.
            fullname = self.distribution.get_fullname()
            installer_name = self.get_installer_filename(fullname)           
            print "changing",installer_name,"to overwrite files on install"
            
            msilib.add_data(self.db, "Property", [("REINSTALLMODE", "amus")])
            self.db.Commit()
    
    cmdclass['bdist_msi'] = bdist_msi_overwrite_on_install


#finally,
#call distutils with all needed info
PACKAGEDATA = {
       "cmdclass":    cmdclass,
       "packages":    ['pygame', 'pygame.gp2x', 'pygame.threads'],
       "package_dir": {'pygame': 'lib',
                       'pygame.threads': 'lib/threads',
                       'pygame.gp2x': 'lib/gp2x'},
       "headers":     headers,
       "ext_modules": extensions,
       "data_files":  [['pygame', data_files]],
}
PACKAGEDATA.update(METADATA)
PACKAGEDATA.update(EXTRAS)
setup(**PACKAGEDATA)
