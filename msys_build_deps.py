#!/usr/bin/env python
# -*- coding: ascii -*-
# Program msys_build_deps.py
# Requires Python 2.4 or later and win32api.

"""Build Pygame dependencies using MinGW and MSYS

Configured for Pygame 1.9.1 and Python 2.5 and up.

The libraries are installed in /usr/local of the MSYS directory structure.

This program can be run from a Windows cmd.exe or MSYS terminal. The current
directory and its outer directory are searched for the library source
directories. Run the program from the pygame trunk directory. The Windows
file path cannot have spaces in it.

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

SDL 1.2.14 
SDL_image 1.2.6
SDL_mixer 1.2 (.8) revision 3942 from SVN
SDL_ttf 2.0.9
smpeg revision 389 from SVN
freetype 2.3.12
libogg 1.1.3
libvorbis 1.2.0
FLAC 1.2.1
tiff 3.8.2
libpng 1.2.43
jpeg 6b
zlib 1.2.3
PortMidi revision 201 from SVN (patched)
ffmpeg revision 23520 from SVN

The build environment used:

gcc-core-4.5.0-1-mingw32
gcc-c++-4.5.0-1-mingw32
libstdc++-4.5.0-1-mingw32
binutils-2.20.1-2-mingw32
mingwrt-3.18-mingw32
w32api-3.14-mingw32
libgmp-5.0.1-1-mingw32
libmpc-0.8.1-1-mingw32
libmpfr-2.4.1-1-mingw32
mingw32-make-3.81-20080326 (needed?)
coreutils-5.97-3-msys-1.0.13
msysDTK-1.0.1 (?)
msys-automake-1.8.2 (?)
autoconf-2.65-1-msys-1.0.13
m4-1.4.14-1-msys-1.0.13
nasm-2.08.01-win32 (needed (replace with yasm)?)
yasm (1.0.1) (Prefered over nasm by SDL 1.2.14 configure?)
plus numerous other dependencies to still be sorted out when the the new
  MinGW installer is available (Hopefully there will be an Msys installer as well.)

The build has been performed on Windows XP.

Build issues:
  An intermitent problem was noted with SDL's configure involving locking of
  conftest.exe resulting in various C library functions being reported unavailable
  when in fact they are present. This does not appear to be a problem with the
  configure script itself but rather Msys. If it happens then just rerun
  msys_build_deps.py.
"""

import msys

from optparse import OptionParser
import os
import sys
from glob import glob
import time

# For Python 2.x/3.x compatibility
def geterror():
    return sys.exc_info()[1]

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

class BuildError(Exception):
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
        except BuildError:
            print_(geterror())
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

class ChooseError(Exception):
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
    except ChooseError:
        print_(geterror())
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
    except msys.MsysException:
        print_(geterror())
        return 1
    start_time = None
    return_code = 1
    set_environment_variables(m, options)
    try:
        configure(chosen_deps)
    except BuildError:
        print_("Build aborted:", geterror())
    else:
        print_("\n=== Starting build ===")
        start_time = time.time()  # For file timestamp checks.
        try:
            build(chosen_deps, m)
        except BuildError:
            print_("Build aborted:", geterror())
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
# bin directory is in PATH. Four other environment variables are defined:
# BDCONF, BDCOMP, BDINST and BDCLEAN. They are either '0' or '1'. They
# represent configure, compile, install and clean respectively. When '1' the
# corresponding action is performed. When '0' it is skipped. A final variable,
# DBWD, is the root directory of the source code. A script will cd to it before
# doing anything else.
#
# None of these scripts end with an "exit". Exit, possibly, leads to Msys
# freezing on some versions of Windows (98).
# 
# The list order corresponds to build order. It is critical.
dependencies = [
    Dependency('SDL', ['SDL-[1-9].*'], ['SDL.dll'], """

set -e
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  # Remove NONAMELESSUNION from directx.h headers.
  for d in video audio; do
    BDDXHDR=src/$d/windx5/directx.h
    cp -f $BDDXHDR $BDDXHDR'_'
    sed 's/^\\(#define NONAMELESSUNION\\)/\\/*\\1*\\//' $BDDXHDR'_' >$BDDXHDR
    if [ x$? != x0 ]; then exit $?; fi
    rm $BDDXHDR'_'
    BDDXHDR=
  done

  # This comes straight from SVN so has no configure script
  if [ ! -f "./configure" ]; then
    ./autogen.sh
  fi
  ./configure
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
  # Make SDL_config_win32.h available for prebuilt and MSVC
  cp -f "$BDWD/include/SDL_config_win32.h" "/usr/local/include/SDL"
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
cd "$BDWD"

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
    Dependency('FREETYPE', ['freetype-[2-9].*'], ['libfreetype-6.dll'], """

set -e
cd "$BDWD"
export PWD="${BDWD// /\\\\ /}"

if [ x$BDCONF == x1 ]; then
  # Need to define inline as freetypes is compiled as -pedentic
  # yet stdlib.h is not.
  export CPPFLAGS="-Dinline=__inline__ $CPPFLAGS"
  ./configure
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/libfreetype-6.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('FONT', ['SDL_ttf-[2-9].*'], ['SDL_ttf.dll'], """

set -e
cd "$BDWD"

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
    Dependency('PNG', ['libpng-[1-9].*'], ['libpng12.dll'], """

set -e
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  # Use the provided MinGW makefile.
  cp -f scripts/Makefile.mingw Makefile
fi

if [ x$BDCOMP == x1 ]; then
  make prefix=/usr/local
fi

if [ x$BDINST == x1 ]; then
  # The makefile goes into an infinite loop on install.
  # Do in ourselves.
  cp -f libpng12.dll /usr/local/bin
  cp -f libpng.dll.a /usr/local/lib
  cp -f libpng.a /usr/local/lib
  headers=/usr/local/include
  mkdir -p $headers
  cp -f png.h $headers
  cp -f pngconf.h $headers
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/libpng12.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('JPEG', ['jpeg-[6-9]*'], ['jpeg.dll'], """

set -e
cd "$BDWD"

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
  cp -fp libjpeg.a /usr/local/lib
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
cd "$BDWD"

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
  # Don't install any libtools info files so SDL_image will not
  # statically link to jpeg.
  cd libtiff
  make install-data-am
  cp -fp .libs/libtiff.a /usr/local/lib
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
cd "$BDWD"

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
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  # This comes straight from SVN so has no configure script
  if [ ! -f "./configure" ]; then
    ./autogen.sh
  fi

  # Don't need the toys.
  ./configure --disable-gtk-player --disable-opengl-player --disable-gtktest
fi

if [ x$BDCOMP == x1 ]; then
  # Leave out undefined symbols so a dll will build, along with the appropriate
  # libtool generated files (if needed elsewere). The DLL is it not what we want
  # though: linked to the C runtime we specify, and statically linked to the gcc
  # runtime libraries (If you know how to coerce libtool into building this DLL
  # then please fix this.)
  make CXXLD='$(CXX) -no-undefined'

  # Build the DLL we want. Build options adapted from those libtool generates in
  # Makefile, but letting g++ do the work of adding support libraries. Note the
  # addition of the -static-libgcc and -static-libstdc++ options.
  g++ -shared -static-libgcc -static-libstdc++ -mwindows \
    -Wl,--enable-auto-image-base -Xlinker --out-implib -Xlinker .libs/libsmpeg.dll.a \
    -L/local/lib -o .libs/smpeg.dll \
    .libs/MPEG.o .libs/MPEGring.o .libs/MPEGlist.o .libs/MPEGstream.o \
    .libs/MPEGsystem.o .libs/MPEGfilter.o .libs/smpeg.o .libs/MPEGaudio.o \
    .libs/bitwindow.o .libs/filter.o .libs/filter_2.o .libs/hufftable.o \
    .libs/mpeglayer1.o .libs/mpeglayer2.o .libs/mpeglayer3.o .libs/mpegtable.o \
    .libs/mpegtoraw.o .libs/MPEGvideo.o .libs/decoders.o .libs/floatdct.o \
    .libs/gdith.o .libs/jrevdct.o .libs/motionvec.o .libs/parseblock.o \
    .libs/readfile.o .libs/util.o .libs/video.o .libs/vhar128.o .libs/mmxflags_asm.o \
    .libs/mmxidct_asm.o -lSDL -lmingw32 -lgcc_s -lgcc -lmoldname -lmingwex \
    -lmsvcrt -luser32 -lkernel32 -ladvapi32 -lshell32
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
cd "$BDWD"

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
cd "$BDWD"

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
    Dependency('FLAC', ['flac-[1-9].*'], [], """

set -e
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  # Add __MINGW32__ to SIZE_T_MAX declaration test in alloc.h header.
  BDHDR='include/share/alloc.h'
  BDTMP='alloc.h_'
  cp -f "$BDHDR" "$BDTMP"
  sed 's/^#  ifdef _MSC_VER$/#  if defined _MSC_VER || defined __MINGW32__/' \
    "$BDTMP" >"$BDHDR"
  rm "$BDTMP"

  # Will only install a static library, but that is all that is needed.
  ./configure --disable-shared --disable-ogg --disable-cpplibs \
    --disable-doxygen-docs
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  cp src/libFLAC/.libs/libFLAC.a /usr/local/lib
  mkdir -p /usr/local/include/FLAC
  cp -f include/FLAC/*.h /usr/local/include/FLAC
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('MIXER', ['SDL_mixer-[1-9].*'], ['SDL_mixer.dll'], """

set -e
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  # This comes straight from SVN so has no configure script
  if [ ! -f "./configure" ]; then
    ./autogen.sh
  fi
  # Add Ws2_32 library for FLAC.
  cp -f configure configure_
  sed '
s/\\(EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lFLAC\\)"/\\1 -lWs2_32"/
s/\\(LIBS="-lFLAC\\)\\(  $LIBS"\\)/\\1 -lWs2_32\\2/' \
    configure_ >configure
  rm configure_

  # No dynamic loading of dependent libraries.
  ./configure --disable-music-ogg-shared --disable-music-mp3-shared \
    --disable-music-flac-shared
fi

if [ x$BDCOMP == x1 ]; then
  # The DLL make builds is it not what we want: linked to the C runtime we specify
  # (If you know how to coerce libtool into building this DLL then please fix this.)
  make

  # Build the DLL we want. Build options adapted from those libtool generates in
  # Makefile, but letting gcc do the work of adding support libraries. Note the
  # addition of the -static-libgcc option.
 gcc -shared -static-libgcc -mwindows-Wl,--enable-auto-image-base \
   -Xlinker --out-implib -Xlinker build/.libs/libSDL_mixer.dll.a \
   -o build/.libs/SDL_mixer.dll -L/usr/local/lib \
   build/.libs/effect_position.o build/.libs/effect_stereoreverse.o \
   build/.libs/effects_internal.o build/.libs/load_aiff.o build/.libs/load_voc.o \
   build/.libs/mixer.o build/.libs/music.o build/.libs/music_cmd.o \
   build/.libs/wavestream.o build/.libs/drv_nos.o build/.libs/load_it.o \
   build/.libs/load_mod.o build/.libs/load_s3m.o build/.libs/load_xm.o \
   build/.libs/mdreg.o build/.libs/mdriver.o build/.libs/mloader.o build/.libs/mlreg.o \
   build/.libs/mlutil.o build/.libs/mmalloc.o build/.libs/mmerror.o \
   build/.libs/mmio.o build/.libs/mplayer.o build/.libs/munitrk.o build/.libs/mwav.o \
   build/.libs/npertab.o build/.libs/sloader.o build/.libs/virtch.o build/.libs/virtch2.o \
   build/.libs/virtch_common.o build/.libs/common.o build/.libs/ctrlmode.o \
   build/.libs/filter.o build/.libs/instrum.o build/.libs/mix.o build/.libs/output.o \
   build/.libs/playmidi.o build/.libs/readmidi.o build/.libs/resample.o \
   build/.libs/sdl_a.o build/.libs/sdl_c.o build/.libs/tables.o build/.libs/timidity.o \
   build/.libs/native_midi_common.o build/.libs/native_midi_mac.o \
   build/.libs/native_midi_win32.o build/.libs/dynamic_ogg.o build/.libs/load_ogg.o \
   build/.libs/music_ogg.o build/.libs/dynamic_flac.o build/.libs/load_flac.o \
   build/.libs/music_flac.o build/.libs/dynamic_mp3.o build/version.o \
   -lwinmm /usr/local/lib/libvorbisfile.dll.a /usr/local/lib/libvorbis.dll.a \
   /usr/local/lib/libogg.dll.a -lFLAC -lWs2_32 /usr/local/lib/libsmpeg.dll.a \
   -lmingw32 -lSDLmain -lSDL
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
    Dependency('PORTMIDI', ['portmidi', 'portmidi-[1-9].*'], ['portmidi.dll'], """

set -e
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  cat > GNUmakefile << 'THE_END'
# Makefile for portmidi, generated for Pygame by msys_build_deps.py.

prefix = /usr/local

pmcom = pm_common
pmwin = pm_win
pt = porttime

pmdll = portmidi.dll
pmlib = libportmidi.a
pmimplib = libportmidi.dll.a
pmcomsrc = $(pmcom)/portmidi.c $(pmcom)/pmutil.c
pmwinsrc = $(pmwin)/pmwin.c $(pmwin)/pmwinmm.c
pmobj = portmidi.o pmutil.o pmwin.o pmwinmm.o
pmsrc = $(pmcomsrc) $(pmwinsrc)
pmreqhdr = $(pmcom)/portmidi.h $(pmcom)/pmutil.h
pmhdr = $(pmreqhdr) $(pmcom)/pminternal.h $(pmwin)/pmwinmm.h

ptsrc = $(pt)/porttime.c porttime/ptwinmm.c
ptobj = porttime.o ptwinmm.o
ptreqhdr = $(pt)/porttime.h
pthdr = $(ptreqhdr)

src = $(pmsrc) $(ptsrc)
reqhdr = $(pmreqhdr) $(ptreqhdr)
hdr = $(pmhdr) $(pthdr)
obj = $(pmobj) $(ptobj)
def = portmidi.def

IHDR := -I$(pmcom) -I$(pmwin) -I$(pt)
LIBS := $(LOADLIBES) $(LDLIBS) -lwinmm



all : $(pmdll)
.PHONY : all

$(pmlib) : $(src) $(hdr)
\tg++ $(CPPFLAGS) $(IHDR) -c $(CFLAGS) $(src)
\tar rc $(pmlib) $(obj)
\tranlib $(pmlib)

$(pmdll) : $(pmlib) $(def)
\tg++ -shared -static-libgcc $(LDFLAGS) -def $(def) $(pmlib) $(LIBS) -o $@
\tdlltool -D $(pmdll) -d $(def) -l $(pmimplib)
\tranlib $(pmimplib)

.PHONY : install

install : $(pmdll)
\tcp -f --target-directory=$(prefix)/bin $<
\tcp -f --target-directory=$(prefix)/lib $(pmlib)
\tcp -f --target-directory=$(prefix)/lib $(pmimplib)
\tcp -f --target-directory=$(prefix)/include $(reqhdr)

.PHONY : clean

clean :
\trm -f $(obj) $(pmdll) $(pmimplib) $(pmlib)
THE_END

  cat > portmidi.def << 'THE_END'
LIBRARY portmidi.dll
EXPORTS
Pm_Abort
Pm_Close
Pm_CountDevices
Pm_Dequeue
Pm_Enqueue
Pm_GetDefaultInputDeviceID
Pm_GetDefaultOutputDeviceID
Pm_GetDeviceInfo
Pm_GetErrorText
Pm_GetHostErrorText
Pm_HasHostError
Pm_Initialize
Pm_OpenInput
Pm_OpenOutput
Pm_Poll
Pm_QueueCreate
Pm_QueueDestroy
Pm_QueueEmpty
Pm_QueueFull
Pm_QueuePeek
Pm_Read
Pm_SetChannelMask
Pm_SetFilter
Pm_SetOverflow
Pm_Terminate
Pm_Write
Pm_WriteShort
Pm_WriteSysEx
Pt_Sleep
Pt_Start
Pt_Started
Pt_Stop
Pt_Time
THE_END

fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/portmidi.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
  rm -f GNUmakefile portmidi.def
fi
"""),
    Dependency('FFMPEG', ['ffmpeg'],
    ['avformat-52.dll', 'swscale-0.dll',
     'avcodec-52.dll', 'avutil-50.dll' ], """

set -e
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  ./configure --enable-shared --enable-memalign-hack
fi

if [ x$BDCOMP == x1 ]; then
  make
  cd libswscale/
  make
  cd ..
fi

if [ x$BDINST == x1 ]; then
  make install
  cd libswscale/
  make install
  cd ..
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all /usr/local/bin/avformat-52.dll
  strip --strip-all /usr/local/bin/swscale-0.dll
  strip --strip-all /usr/local/bin/avcodec-52.dll
  strip --strip-all /usr/local/bin/avutil-50.dll
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
  cd libswscale/
  make clean
  cd ..
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
  mkdir -p "$DBMSVCR71"
  cp -fp /mingw/lib/libmoldname71.a "$DBMSVCR71/libmoldname.a"
  cp -fp /mingw/lib/libmoldname71d.a "$DBMSVCR71/libmoldnamed.a"
  cp -fp /mingw/lib/libmsvcr71.a "$DBMSVCR71/libmsvcrt.a"
  cp -fp /mingw/lib/libmsvcr71d.a "$DBMSVCR71/libmsvcrtd.a"
fi
""")

if __name__ == '__main__':
    sys.exit(main(dependencies, msvcr71_prep, msys_prep))
