#!/usr/bin/env python
# -*- coding: ascii -*-
# Program msys_build_deps.py
# Requires Python 2.4 or later and win32api.

"""Build Pygame dependencies using MinGW and MSYS

Configured for Pygame 1.8 and Python 2.4 and up.

The libraries are installed in /usr/local of the MSYS directory structure.

This program can be run from a Windows cmd.exe or MSYS terminal. The current
directory and its outer directory are searched for the library source
directories.

The recognized, and optional, environment variables are:
  SHELL - MSYS shell program path - already defined in the MSYS terminal
  CFLAGS - compiler options - overrides the defaults used by this program
  LDFLAGS - linker options - prepended to flags set by the program
  LIBRARY_PATH - library directory paths - appended to those used by this
                 program
  CPATH - C/C++ header file paths - appended to the paths used by this program

To get a list of command line options run

python build_deps.py --help

This program has been tested against the following libraries:

SDL 1.2.13
SDL_image 1.2.6
SDL_mixer 1.2.8
SDL_ttf 2.0.9
smpeg revision 370 from SVN
freetype 2.3.5
libogg 1.1.3
libvorbis 1.2.0
tiff 3.8.2
libpng 1.2.24
jpeg 6b
zlib 1.2.3

The build environment used:

gcc-core-3.4.5
binutils-2.17.50
mingw-runtime-3.14
win32api-3.11
mingw32-make-3.81.2
gcc-g++3.4.5
MSYS-1.0.11
MSYS-DTK-1.0.1 (for smpeg)
nasm (from www.libsdl.org)

Builds have been performed on Windows 98 and XP.

Build issues:
  For pre-2007 computers:  MSYS bug "[ 1170716 ] executing a shell scripts
    gives a memory leak" (http://sourceforge.net/tracker/
    index.php?func=detail&aid=1170716&group_id=2435&atid=102435)
    
    It may not be possible to use the --all option to build all Pygame
    dependencies in one session. Instead the job may need to be split into two
    or more sessions, with a reboot of the operatingsystem between each. Use
    the --help-args option to list the libraries in the their proper build
    order.
"""

import msys

from optparse import OptionParser
import os
import sys
from glob import glob
import time

#
#   Generic declarations
#
hunt_paths = ['.', '..']

def prompt(p=None):
    """MSYS friendly raw_input
    
    This provides a hook that can be replaced for testing.
    """
    
    msys.msys_raw_input(p)

def print_(*args, **kwds):
    msys.msys_print(*args, **kwds)

def confirm(message):
    """Ask a yes/no question, return result"""
    
    reply = prompt("\n%s [Y/n]:" % message)
    if reply and reply[0].lower() == 'n':
        return 0
    return 1

def as_flag(b):
    """Return bool b as a shell script flag '1' or '0'"""
    
    if b:
        return '1'
    return '0'

def merge_strings(*args, **kwds):
    """Returns non empty string joined by sep

    The default separator is an empty string.
    """

    sep = kwds.get('sep', '')
    return sep.join([s for s in args if s])

class BuildError(StandardError):
    """Raised for missing source paths and failed script runs"""
    pass

class Dependency(object):
    """Builds a library"""
    
    def __init__(self, name, wildcards, dlls, shell_script):
        self.name = name
        self.wildcards = wildcards
        self.shell_script = shell_script
        self.dlls = dlls

    def configure(self, hunt_paths):
        self.path = None
        self.paths = []
        self.hunt(hunt_paths)
        self.choosepath()

    def hunt(self, hunt_paths):
        parent = os.path.abspath('..')
        for p in hunt_paths:
            for w in self.wildcards:
                found = glob(os.path.join(p, w))
                found.sort() or found.reverse()  #reverse sort
                for f in found:
                    if f[:5] == '..'+os.sep+'..' and \
                        os.path.abspath(f)[:len(parent)] == parent:
                        continue
                    if os.path.isdir(f):
                        self.paths.append(f)

    def choosepath(self):
        path = None
        if not self.paths:
            raise BuildError("Path for %s: not found" % self.name)
        if len(self.paths) == 1:
            path = self.paths[0]
        else:
            print_("Select path for %s:" % self.name)
            for i in range(len(self.paths)):
                print_("  %d = %s" % (i+1, self.paths[i]))
            print_("  0 = <Nothing>")
            choice = prompt("Select 0-%d (1=default):" % len(self.paths))
            if not choice:
                choice = 1
            else:
                choice = int(choice)
            if choice > 0:
                path = self.paths[choice-1]
        if path is not None:
            self.path = os.path.abspath(path)

    def build(self, msys):
        if self.path is not None:
            msys.environ['BDWD'] = msys.windows_to_msys(self.path)
            return_code = msys.run_shell_script(self.shell_script)
            if return_code != 0:
                raise BuildError("The build for %s failed with code %d" %
                                 (self.name, return_code))
        else:
            raise BuildError("No source directory for %s" % self.name)

class Preparation(object):
    """Perform necessary build environment preperations"""
    
    def __init__(self, name, shell_script):
        self.name = name
        self.path = ''
        self.paths = []
        self.dlls = []
        self.shell_script = shell_script

    def configure(self, hunt_paths):
        pass

    def build(self, msys):
        return_code = msys.run_shell_script(self.shell_script)
        if return_code != 0:
            raise BuildError("Preparation '%s' failed with code %d" %
                             (self.name, return_code))

def configure(dependencies):
    """Find source directories of all dependencies"""
    
    success = True
    print_("Hunting for source directories...")
    for dep in dependencies:
        try:
            dep.configure(hunt_paths)
        except BuildError, e:
            print_(e)
            success = False
        else:
            if dep.path:
                print_("Source directory for", dep.name, ":", dep.path)
    if not success:
        raise BuildError("Not all source directories were found")

def build(dependencies, msys):
    """Execute that shell scripts for all dependencies"""
    
    for dep in dependencies:
        dep.build(msys)

def command_line():
    """Process the command line and return the options"""
    
    usage = ("usage: %prog [options] --all\n"
             "       %prog [options] [args]\n"
             "\n"
             "Build the Pygame dependencies. The args, if given, are\n"
             "libraries to include or exclude.\n"
             "\n"
             "At startup this program may prompt for missing information.\n"
             "Be aware of this before redirecting output or leaving the\n"
             "program unattended. Once the 'Starting build' message appears\n"
             "no more user input is required. The build process will"
             "abort on the first error, as library build order is important.\n"
             "\n"
             "See --include and --help-args.\n"
             "\n"
             "For more details see the program's document string\n")
    
    parser = OptionParser(usage)
    parser.add_option('-a', '--all', action='store_true', dest='build_all',
                      help="Include all libraries in the build")
    parser.set_defaults(build_all=False)
    parser.add_option('--no-msvcr71', action='store_false', dest='msvcr71',
                      help="Do not link to msvcr71.dll")
    parser.set_defaults(msvcr71=True)
    parser.add_option('--console', action='store_true', dest='console',
                      help="Link with the console subsystem:"
                           " defaults to Win32 GUI")
    parser.set_defaults(console=False)
    parser.add_option('--no-configure', action='store_false', dest='configure',
                      help="Do not prepare the makefiles")
    parser.set_defaults(configure=True)
    parser.add_option('--no-compile', action='store_false', dest='compile',
                      help="Do not compile or install the libraries")
    parser.set_defaults(compile=True)
    parser.add_option('--no-install', action='store_false', dest='install',
                      help="Do not install the libraries")
    parser.add_option('--no-strip', action='store_false', dest='strip',
                      help="Do not strip the library")
    parser.set_defaults(strip=True)
    parser.set_defaults(install=True)
    parser.add_option('--clean', action='store_true', dest='clean',
                      help="Remove generated files (make clean)"
                           " as a last step")
    parser.set_defaults(clean=False)
    parser.add_option('--clean-only', action='store_true', dest='clean_only',
                      help="Perform only a clean")
    parser.set_defaults(clean_only=False)
    parser.add_option('-e', '--exclude', action='store_true', dest='exclude',
                      help="Exclude the specified libraries")
    parser.set_defaults(exclude=False)
    parser.add_option('-m', '--msys-root', action='store',
                      dest='msys_directory',
                      help="MSYS directory path, which may include"
                           " the 1.x subdirectory")
    parser.set_defaults(msys_directory='')
    parser.add_option('--help-args', action='store_true', dest='arg_help',
                      help="Show a list of recognised libraries,"
                           " in build order, and exit")
    parser.set_defaults(arg_help=False)
    return parser.parse_args()

def set_environment_variables(msys, options):
    """Set the environment variables used by the scripts"""
    
    environ = msys.environ
    msys_root = msys.msys_root
    environ['BDCONF'] = as_flag(options.configure and
                                not options.clean_only)
    environ['BDCOMP'] = as_flag(options.compile and
                                not options.clean_only)
    environ['BDINST'] = as_flag(options.install and
                                options.compile and
                                not options.clean_only)
    environ['BDSTRIP'] = as_flag(options.install and
                                 options.strip and
                                 not options.clean_only)
    environ['BDCLEAN'] = as_flag(options.clean or options.clean_only)
    environ.pop('INCLUDE', None)  # INCLUDE causes problems with MIXER.
    if 'CFLAGS' not in environ:
        environ['CFLAGS'] = '-O2'
    ldflags = '-mwindows'
    if options.console:
        ldflags = '-mconsole'
    environ['LDFLAGS'] = merge_strings(environ.get('LDFLAGS', ''), ldflags,
                                       sep=' ')
    library_path = os.path.join(msys_root, 'local', 'lib')
    msvcr71_path = ''
    if options.msvcr71:
        # Hide the msvcrt.dll import libraries with those for msvcr71.dll.
        # Their subdirectory is in the same directory as the SDL library.
        msvcr71_path = os.path.join(library_path, 'msvcr71')
        environ['DBMSVCR71'] = msvcr71_path
    # For dependency libraries and msvcrt hiding.
    environ['LIBRARY_PATH'] = merge_strings(library_path, msvcr71_path,
                                            environ.get('LIBRARY_PATH', ''),
                                            sep=';')
    # For dependency headers.
    include_path = os.path.join(msys_root, 'local', 'include')
    environ['CPATH'] = merge_strings(include_path, environ.get('CPATH', ''),
                                     sep=';')

class ChooseError(StandardError):
    """Failer to select dependencies"""
    pass

def choose_dependencies(dependencies, options, args):
    """Return the dependencies to actually build"""

    if options.build_all:
        if args:
            raise ChooseError("No library names are accepted"
                              " for the --all option.")
        if options.exclude:
            return []
        else:
            return dependencies

    if args:
        names = [d.name for d in dependencies]
        args = [a.upper() for a in args]
        for a in args:
            if a not in names:
                msg = ["%s is an unknown library; valid choices are:" % a]
                msg.extend(names)
                raise ChooseError('\n'.join(msg))
        if options.exclude:
            return [d for d in dependencies if d.name not in args]
        return [d for d in dependencies if d.name in args]

    return []
    
def summary(dependencies, msys, start_time, chosen_deps):
    """Display a summary report of new, existing and missing DLLs"""

    import datetime

    print_("\n\n=== Summary ===")
    if start_time is not None:
        print_("  Elapse time:",
               datetime.timedelta(seconds=time.time()-start_time))
    print_()
    for dep in chosen_deps:
        if dep.path is None:
            print_("  ** No source directory found for", dep.name)
        elif dep.path:
            print_("  Source directory for", dep.name, ":", dep.path)
    print_()
    msys_root = msys.msys_root
    bin_dir = os.path.join(msys_root, 'local', 'bin')
    for d in dependencies:
        name = d.name
        dlls = d.dlls
        for dll in dlls:
            dll_path = os.path.join(bin_dir, dll)
            try:
                mod_time = os.path.getmtime(dll_path)
            except:
                msg = "No DLL"
            else:
                if mod_time >= start_time:
                    msg = "Installed new DLL %s" % dll_path
                else:
                    msg = "-- (old DLL %s)" % dll_path
            print_("  %-10s: %s" % (name, msg))
    
def main(dependencies, msvcr71_preparation, msys_preparation):
    """Build the dependencies according to the command line options."""

    options, args = command_line()
    if options.arg_help:
        print_("These are the Pygame library dependencies:")
        for dep in dependencies:
            print_(" ", dep.name)
        return 0
    try:
        chosen_deps = choose_dependencies(dependencies, options, args)
    except ChooseError, e:
        print_(e)
        return 1
    if not chosen_deps:
        if not args:
            print_("No libraries specified.")
        elif options.build_all:
            print_("All libraries excluded")
    if options.msvcr71:
        chosen_deps.insert(0, msvcr71_preparation)
    if chosen_deps:
        chosen_deps.insert(0, msys_preparation)
    try:
        m = msys.Msys(options.msys_directory)
    except msys.MsysException, e:
        print_(e)
        return 1
    start_time = None
    return_code = 1
    set_environment_variables(m, options)
    try:
        configure(chosen_deps)
    except BuildError, e:
        print_("Build aborted:", e)
    else:
        print_("\n=== Starting build ===")
        start_time = time.time()  # For file timestamp checks.
        try:
            build(chosen_deps, m)
        except BuildError, e:
            print_("Build aborted:", e)
        else:
            # A successful build!
            return_code = 0
    summary(dependencies, m, start_time, chosen_deps)

    # MinGW configure file for setup.py (optional).
    try:
        import mingwcfg
    except ImportError:
        pass
    else:
        mingwcfg.write(m.mingw_root)

    return return_code

#
#   Build specific code
#

# This list includes the MSYS shell scripts to build each library. Each script
# runs in an environment where MINGW_ROOT_DIRECTORY is defined and the MinGW
# bin directory is in PATH. Three other environment variables are defined:
# BDCONF, BDCOMP, BDINST and BDCLEAN. They are either '0' or '1'. They
# represent configure, compile, install and clean respectively. When '1' the
# corresponding action is performed. When '0' it is skipped. A final variable,
# DBWD, is the root directory of the source code. A script will cd to it before
# doing anything else.
# 
# The list order corresponds to build order. It is critical.
dependencies = [
    Dependency('SDL', ['SDL-[1-9].*'], ['SDL.dll'], """

set -e
cd $BDWD

if [ x$BDCONF == x1 ]; then
  # Remove NONAMELESSUNION from directx.h headers.
  for d in video audio; do
    BDDXHDR=src/$d/windx5/directx.h
    cp -f $BDDXHDR $BDDXHDR'_'
    sed 's/^\\(\\#define NONAMELESSUNION\\)/\\/*\\1*\\//' $BDDXHDR'_' >$BDDXHDR
    if [ x$? != x0 ]; then exit $?; fi
    rm $BDDXHDR'_'
    BDDXHDR=
  done

  ./configure
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/SDL.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('Z', ['zlib-[1-9].*'], ['zlib1.dll'], """

set -e
cd $BDWD

if [ x$BDCONF == x1 ]; then
  # Use the existing gcc makefile, modified to add linker options.
  sed "s/dllwrap/dllwrap $LDFLAGS/" win32/Makefile.gcc >Makefile.gcc
fi

if [ x$BDCOMP == x1 ]; then
  # Build with the import library renamed.
  make IMPLIB='libz.dll.a' -fMakefile.gcc "CFLAGS=$CFLAGS"
fi

if [ x$BDINST == x1 ]; then
  # Have to do own install.
  cp -fp *.a /usr/local/lib
  cp -fp zlib.h /usr/local/include
  cp -fp zconf.h /usr/local/include
  cp -fp zlib1.dll /usr/local/bin
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/zlib1.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('FREETYPE', ['freetype-[2-9].*'], [], """

set -e
cd $BDWD

if [ x$BDCONF == x1 ]; then
  # Will only install a static library, but that is all that is needed.
  ./configure --disable-shared
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('FONT', ['SDL_ttf-[2-9].*'], ['SDL_ttf.dll'], """

set -e
cd $BDWD

if [ x$BDCONF == x1 ]; then
  ./configure
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/SDL_ttf.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('PNG', ['libpng-[1-9].*'], ['libpng13.dll'], """

set -e
cd $BDWD

if [ x$BDCONF == x1 ]; then
  # This will only build a static library.
  ./configure --with-libpng-compat=no --disable-shared

  # Remove a duplicate entry in the def file.
  sed '222 d' scripts/pngw32.def >in.def
fi

if [ x$BDCOMP == x1 ]; then
  # Build the DLL as win32 gui.
  make
  dlltool -D libpng13.dll -d in.def -l libpng.dll.a
  ranlib libpng.dll.a
  gcc -shared -s $LDFLAGS -def in.def -o libpng13.dll .libs/libpng12.a -lz
fi

if [ x$BDINST == x1 ]; then
  # Only install the headers and import library, otherwise SDL_image will
  # statically link to png.
  make install-pkgincludeHEADERS
  cp -fp png.h /usr/local/include
  cp -fp pngconf.h /usr/local/include
  cp -fp libpng.dll.a /usr/local/lib
  cp -fp libpng13.dll /usr/local/bin
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/libpng13.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
  rm -f in.def
  rm -f libpng.dll.a
  rm -f libpng13.dll
fi
"""),
    Dependency('JPEG', ['jpeg-[6-9]*'], ['jpeg.dll'], """

set -e
cd $BDWD

if [ x$BDCONF == x1 ]; then
  # This will only build a static library.
  ./configure --disable-shared
fi

if [ x$BDCOMP == x1 ]; then
  # Build the DLL as a win32 gui.
  make
  dlltool --export-all-symbols -D jpeg.dll -l libjpeg.dll.a -z in.def libjpeg.a
  ranlib libjpeg.dll.a
  gcc -shared -s $LDFLAGS -def in.def -o jpeg.dll libjpeg.a
fi

if [ x$BDINST == x1 ]; then
  # Only install the headers and import library, otherwise SDL_image will
  # statically link to jpeg.
  make install-headers
  cp -fp libjpeg.dll.a /usr/local/lib
  cp -fp jpeg.dll /usr/local/bin
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/jpeg.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
  rm -f in.def
  rm -f libjpeg.dll.a
  rm -f jpeg.dll
fi
"""),
    Dependency('TIFF', ['tiff-[3-9].*'], ['libtiff.dll'], """

set -e
cd $BDWD

if [ x$BDCONF == x1 ]; then
  # The shared library build does not work
  ./configure --disable-cxx --prefix=/usr/local --disable-shared
fi

if [ x$BDCOMP == x1 ]; then
  make

  # Build the DLL as a win32 gui
  cd libtiff
  gcc -shared -s $LDFLAGS -def libtiff.def -o libtiff.dll .libs/libtiff.a \
    -ljpeg -lz
  dlltool -D libtiff.dll -d libtiff.def -l libtiff.dll.a
  ranlib libtiff.dll.a
  cd ..
fi

if [ x$BDINST == x1 ]; then
  # Only install the headers and import library, otherwise SDL_image will
  # statically link to jpeg.
  cd libtiff
  make install-data-am
  cp -fp libtiff.dll.a /usr/local/lib
  cp -fp libtiff.dll /usr/local/bin
  if [ x$? != x0 ]; then exit $?; fi
  cd ..
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/libtiff.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
  rm -f libtiff/libtiff.dll.a
  rm -f libtiff/libtiff.dll
fi
"""),
    Dependency('IMAGE', ['SDL_image-[1-9].*'], ['SDL_image.dll'], """

set -e
cd $BDWD

if [ x$BDCONF == x1 ]; then
  # Disable dynamic loading of image libraries as that uses the wrong DLL
  # search path
  ./configure --disable-jpeg-shared --disable-png-shared --disable-tif-shared
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/SDL_image.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('SMPEG', ['smpeg-[0-9].*', 'smpeg'], ['smpeg.dll'], """

set -e
cd $BDWD

if [ x$BDCONF == x1 ]; then
  # This comes straight from SVN so has no configure script
  ./autogen.sh
  ./configure --disable-gtk-player --disable-opengl-player --disable-gtktest
fi

if [ x$BDCOMP == x1 ]; then
  make CXXLD='$(CXX) -no-undefined'
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/smpeg.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('OGG', ['libogg-[1-9].*'], ['libogg-0.dll'], """

set -e
cd $BDWD

if [ x$BDCONF == x1 ]; then
  ./configure
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/libogg-0.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('VORBIS',
               ['libvorbis-[1-9].*'],
               ['libvorbis-0.dll', 'libvorbisfile-3.dll'], """

set -e
cd $BDWD

if [ x$BDCONF == x1 ]; then
  ./configure
fi

if [ x$BDCOMP == x1 ]; then
  make LIBS='-logg'
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/libvorbis-0.dll
  strip --strip-all /usr/local/bin/libvorbisfile-3.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('MIXER', ['SDL_mixer-[1-9].*'], ['SDL_mixer.dll'], """

set -e
cd $BDWD

if [ x$BDCONF == x1 ]; then
  ./configure --disable-music-ogg-shared --disable-music-mp3-shared
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/SDL_mixer.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    ]  # End dependencies = [.


msys_prep = Preparation('/usr/local', """

# Ensure /usr/local and its subdirectories exist.
mkdir -p /usr/local/lib
mkdir -p /usr/local/include
mkdir -p /usr/local/bin
mkdir -p /usr/local/doc
mkdir -p /usr/local/man
mkdir -p /usr/local/share
""")
    
msvcr71_prep = Preparation('msvcr71.dll linkage', r"""

set -e

#
#   msvcr71.dll support
#
if [ ! -f "$DBMSVCR71/libmoldname.a" ]; then
  OBJS='isascii.o iscsym.o iscsymf.o toascii.o
        strcasecmp.o strncasecmp.o wcscmpi.o'
  if [ ! -d /tmp/build_deps ]; then mkdir /tmp/build_deps; fi
  cd /tmp/build_deps
  # These definitions are taken from mingw-runtime-3.12 .
  # The file was generated with the following command:
  #
  # gcc -DRUNTIME=msvcrt -D__FILENAME__=moldname-msvcrt.def
  #   -D__MSVCRT__ -C -E -P -xc-header moldname.def.in >moldname-msvcrt.def
  cat > moldname-msvcrt.def << 'THE_END'
EXPORTS
access
chdir
chmod
chsize
close
creat
cwait

daylight DATA

dup
dup2
ecvt
eof
execl
execle
execlp
execlpe
execv
execve
execvp
execvpe
fcvt
fdopen
fgetchar
fgetwchar
filelength
fileno
; Alias fpreset is set in CRT_fp10,c and CRT_fp8.c.
; fpreset
fputchar
fputwchar
fstat
ftime
gcvt
getch
getche
getcwd
getpid
getw
heapwalk
isatty
itoa
kbhit
lfind
lsearch
lseek
ltoa
memccpy
memicmp
mkdir
mktemp
open
pclose
popen
putch
putenv
putw
read
rmdir
rmtmp
searchenv
setmode
sopen
spawnl
spawnle
spawnlp
spawnlpe
spawnv
spawnve
spawnvp
spawnvpe
stat
strcmpi
strdup
stricmp
stricoll
strlwr
strnicmp
strnset
strrev
strset
strupr
swab
tell
tempnam

timezone DATA

; export tzname for both. See <time.h>
tzname DATA
tzset
umask
ungetch
unlink
utime
wcsdup
wcsicmp
wcsicoll
wcslwr
wcsnicmp
wcsnset
wcsrev
wcsset
wcsupr

wpopen

write
; non-ANSI functions declared in math.h
j0
j1
jn
y0
y1
yn
chgsign
scalb
finite
fpclass
; C99 functions
cabs
hypot
logb
nextafter
THE_END

  mkdir -p "$DBMSVCR71"
  ar x /mingw/lib/libmoldname.a $OBJS
  dlltool --as as -k -U \
     --dllname msvcr71.dll \
     --def moldname-msvcrt.def \
     --output-lib libmoldname.a
  ar rc libmoldname.a $OBJS
  ranlib libmoldname.a
  mv -f libmoldname.a "$DBMSVCR71"
  ar x /mingw/lib/libmoldnamed.a $OBJS
  dlltool --as as -k -U \
     --dllname msvcr71.dll \
     --def moldname-msvcrt.def \
     --output-lib libmoldnamed.a
  ar rc libmoldnamed.a $OBJS
  ranlib libmoldnamed.a
  mv -f libmoldnamed.a "$DBMSVCR71"
  cp -fp /mingw/lib/libmsvcr71.a "$DBMSVCR71/libmsvcrt.a"
  cp -fp /mingw/lib/libmsvcr71d.a "$DBMSVCR71/libmsvcrtd.a"
  rm -f ./*
  cd $OLDPWD
  rmdir /tmp/build_deps
fi
""")

if __name__ == '__main__':
    sys.exit(main(dependencies, msvcr71_prep, msys_prep))
