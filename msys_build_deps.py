#!/usr/bin/env python
# -*- coding: ascii -*-
# Program msys_build_deps.py
# Requires Python 2.5 or later and win32api.

"""Build Pygame dependencies using MinGW and MSYS

Configured for Pygame 1.9.2 and Python 2.5 and up.

By default the libraries are installed in the MSYS directory /usr/local unless
a diffrent directory is specified by the --prefix command line argument.

This program can be run from a Windows cmd.exe or MSYS terminal. The current
directory and its outer directory are searched for the library source
directories. Run the program from the pygame trunk directory. The Windows
file path cannot have spaces in it.

The recognized, and optional, environment variables are:
  PREFIX - Destination directory
  MSYS_ROOT_DIRECTORY - MSYS home directory (may omit 1.0 subdirectory)
  CPPFLAGS - preprocessor options, appended to options set by the program
  LDFLAGS - linker options - prepended to flags set by the program
  CPATH - C/C++ header file paths - appended to the paths used by this program

To get a list of command line options run

python build_deps.py --help

This program has been tested against the following libraries:

SDL 1.2(.14+) hg changeset c5d651a8b679
 SDL_image 1.2(.10+) hg changset 45748e6e2f81
  SDL_mixer 1.2.11 and revision 6ed75d34edc9 tip from hg
  SDL_ttf 2.0.9
smpeg SVN revision 391
freetype 2.4.8
  libogg 1.2.0
  libvorbis 1.3.1
  FLAC 1.2.1
  mikmod 3.1.12 patched (included with SDL_mixer 1.2.11)
tiff 3.9.4
libpng 1.6.0b1
jpeg 8c
zlib 1.2.5
  PortMidi revision 201 from SVN
  ffmpeg revision 24482 from SVN (swscale revision 31785)

The build environment used: 

GCC 4.6.1
MSYS 1.0.17
dx7 headers
yasm 1.2.0


The build has been performed on Windows XP, SP3.

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

default_prefix_mp = '/usr/local'

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

def as_linker_lib_path(p):
    """Return as an ld library path argument"""
    
    if p:
        return '-L' + p
    return ''

def as_preprocessor_header_path(p):
    """Return as a C preprocessor header include path argument"""
    
    if p:
        return '-I' + p
    return ''

def merge_strings(*args, **kwds):
    """Returns non empty string joined by sep

    The default separator is an empty string.
    """

    sep = kwds.get('sep', '')
    return sep.join([s for s in args if s])

def get_python_msvcr_version():
    """Return the Visual C runtime version Python is linked to, as an int"""
    
    python_version = sys.version_info[0:2]
    if python_version < (2.4):
        return 60
    if python_version < (2.6):
        return 71
    return 90
    
class BuildError(Exception):
    """Raised for missing source paths and failed script runs"""
    pass

class Dependency(object):
    """Builds a library"""
    
    def __init__(self, name, wildcards, libs, shell_script):
        self.name = name
        self.wildcards = wildcards
        self.shell_script = shell_script
        self.libs = libs

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
        self.libs = []
        self.shell_script = shell_script

    def configure(self, hunt_paths):
        pass

    def build(self, msys):
        return_code = msys.run_shell_script(self.shell_script)
        if return_code != 0:
            raise BuildError("Preparation '%s' failed with code %d" %
                             (self.name, return_code))

def configure(dependencies, hunt_paths):
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
    """Execute the shell scripts for all dependencies"""
    
    for dep in dependencies:
        print_("\n\n----", dep.name, "----")
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
             "See --help-args.\n"
             "\n"
             "For more details see the program's document string\n")
    
    parser = OptionParser(usage)
    parser.add_option('-a', '--all', action='store_true', dest='build_all',
                      help="Include all libraries in the build")
    parser.set_defaults(build_all=False)
    parser.add_option('--msvcr-version', action='store', dest='msvcr_version',
                      type='choice', choices=['60', '71', '90'],
                      help="Visual C runtime library version")
    parser.set_defaults(msvcr_version=get_python_msvcr_version())
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
    parser.add_option('-s', '--sources', action='store',
                      dest='sources',
                      help="Paths to search for library source directories"
                           " as a semicolon ';' separated list: defaults to %s"
                           % (';'.join(hunt_paths),))
    parser.add_option('-p', '--prefix', action='store',
                      dest='prefix',
                      help="Destination directory of the build: defaults to MSYS %s"
                           % (default_prefix_mp,))
    parser.set_defaults(prefix='')
    parser.add_option('--help-args', action='store_true', dest='arg_help',
                      help="Show a list of recognised libraries,"
                           " in build order, and exit")
    parser.set_defaults(arg_help=False)
    parser.add_option('--subsystem-noforce', action='store_true', dest='subsystem_noforce',
                      help="Do not force the dlls to build with the GUI subsystem type")
    parser.set_defaults(subsystem_noforce=False)
    parser.add_option('-b', '--beep', action='store_true', dest='finish_alert',
                      help="Beep the computer speaker when finished.")
    parser.set_defaults(finish_alert=False)
    parser.add_option('-n', '--beep-ntimes', type='int', action='store', dest='finish_alert_ntimes',
                      help="Beep the computer speaker n times when finished")
    parser.set_defaults(finish_alert_ntimes=0)
    return parser.parse_args()

def set_environment_variables(msys, options):
    """Set the environment variables used by the scripts"""
    
    environ = msys.environ
    msys_root_wp = msys.msys_root
    prefix_wp = options.prefix
    if not prefix_wp:
        prefix_wp = environ.get('PREFIX', '')
    if prefix_wp:
        prefix_mp = msys.windows_to_msys(prefix_wp)
    else:
        prefix_mp = default_prefix_mp
        prefix_wp = msys.msys_to_windows(prefix_mp)
    environ['PREFIX'] = prefix_mp
    environ['BDCONF'] = as_flag(options.configure and
                                not options.clean_only)
    environ['BDCOMP'] = as_flag(options.compile and
                                not options.clean_only)
    environ['BDINST'] = as_flag(options.install and
                                options.compile and
                                not options.clean_only)
    environ['BDSTRIP'] = as_flag(options.compile and
                                 options.install and
                                 options.strip and
                                 not options.clean_only)
    environ['BDCLEAN'] = as_flag(options.clean or options.clean_only)
    environ.pop('INCLUDE', None)  # INCLUDE causes problems with MIXER.
    lib_mp = prefix_mp + '/lib'
    msvcr_mp = ''
    resources_mp = ''
    if options.msvcr_version == '71':
        # Hide the msvcrt.dll import libraries with those for msvcr71.dll.
        # Their subdirectory is in the same directory as the SDL library.
        msvcr_mp = lib_mp + '/msvcr71'
        environ['BDMSVCR71'] = msvcr_mp
    elif options.msvcr_version == '90':
        # Hide the msvcrt.dll import libraries with those for msvcr90.dll.
        # Their subdirectory is in the same directory as the SDL library.
        msvcr_mp = lib_mp + '/msvcr90'
        environ['BDMSVCR90'] = msvcr_mp
        resources_mp = msvcr_mp + '/resources.o'
        environ['BDRESOURCES'] = resources_mp
    if prefix_wp:
        environ['CPPFLAGS'] = merge_strings(as_preprocessor_header_path(prefix_mp + '/include'),
                                            environ.get('CPPFLAGS', ''),
                                            sep=' ')
    subsystem = ''
    if not options.subsystem_noforce:
        subsystem = '-mwindows'
    environ['LDFLAGS'] = merge_strings(environ.get('LDFLAGS', ''),
                                       as_linker_lib_path(lib_mp),
                                       as_linker_lib_path(msvcr_mp),
                                       resources_mp,
                                       subsystem,
                                       sep=' ')

    # For dependency headers.
    include_wp = prefix_wp + '/include'
    environ['CPATH'] = merge_strings(include_wp, environ.get('CPATH', ''),
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
    """Display a summary report of new, existing and missing libraries"""

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
    prefix = msys.msys_to_windows(msys.environ['PREFIX']).replace('/', os.sep)
    bin_dir = os.path.join(prefix, 'bin')
    lib_dir = os.path.join(prefix, 'lib')
    for d in dependencies:
        for lib in d.libs:
            if lib.endswith('.dll'):
                lib_path = os.path.join(bin_dir, lib)
                try:
                    mod_time = os.path.getmtime(lib_path)
                except:
                    msg = "No DLL"
                else:
                    if mod_time >= start_time:
                        msg = "Installed new DLL %s" % (lib_path,)
                    else:
                        msg = "-- (old DLL %s)" % (lib_path,)
            elif lib.endswith('.a'):
                lib_path = os.path.join(lib_dir, lib)
                try:
                    mod_time = os.path.getmtime(lib_path)
                except:
                    msg = "No static library"
                else:
                    if mod_time >= start_time:
                        msg = "Installed new static library %s" % (lib_path,)
                    else:
                        msg = "-- (old static library %s)" % (lib_path,)
            else:
                msg = "Internal error: unknown library type %s" % (lib,)
            print_("  %-10s: %s" % (d.name, msg))
    
def main(dependencies, msvcr71_preparation, msvcr90_preparation, msys_preparation):
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
    if options.msvcr_version == '71' and not options.clean_only:
        chosen_deps.insert(0, msvcr71_preparation)
        print_("Linking to msvcr71.dll.")
    elif options.msvcr_version == '90' and not options.clean_only:
        chosen_deps.insert(0, msvcr90_preparation)
    else:
        print_("Linking to C runtime library msvcrt.dll.")
    if chosen_deps and not options.clean_only:
        chosen_deps.insert(0, msys_preparation)
    try:
        m = msys.Msys(options.msys_directory)
    except msys.MsysException:
        print_(geterror())
        return 1
    print_("Using MSYS in directory:", m.msys_root)
    print_("MinGW directory:", m.mingw_root)
    start_time = None
    return_code = 1
    set_environment_variables(m, options)
    if not options.clean_only:
        print_("Destination directory:",
               m.msys_to_windows(m.environ['PREFIX']).replace('/', os.sep))
    print_("common CPPFLAGS:", m.environ.get('CPPFLAGS', ''))
    print_("common CFLAGS:", m.environ.get('CFLAGS', ''))
    print_("common LDFLAGS:", m.environ.get('LDFLAGS', ''))
    sources = hunt_paths
    if options.sources:
        sources = options.sources.split(';')
    print_("library source directories search paths: %s" % (';'.join(sources),))
    try:
        configure(chosen_deps, sources)
    except BuildError:
        print_("Build aborted:", geterror())
    else:
        if options.clean_only:
            print_("\n=== Performing clean ===")
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
    if not options.clean_only:
        summary(dependencies, m, start_time, chosen_deps)

    # MinGW configure file for setup.py (optional).
    try:
        import mingwcfg
    except ImportError:
        pass
    else:
        mingwcfg.write(m.mingw_root)

    if options.finish_alert or options.finish_alert_ntimes > 0:
        if options.finish_alert_ntimes > 0:
            m.environ['BDNTIMES'] = "%i" % (options.finish_alert_ntimes,)
        alert.build(m)
    return return_code

#
#   Build specific code
#

# This list includes the MSYS shell scripts to build each library. Each script
# runs in an environment where MINGW_ROOT_DIRECTORY is defined and the MinGW
# bin directory is in PATH. Four build control environment variables are
# defined: BDCONF, BDCOMP, BDINST and BDCLEAN. They are either '0' or '1'. They
# represent configure, compile, install and clean respectively. When '1' the
# corresponding action is performed. When '0' it is skipped. The installation
# directory is given by PREFIX. The script needs to prepend it to PATH. The
# source code root directory is BDWD. A script will cd to it before doing
# starting the build. Various gcc flags are in CPATH, CPPFLAGS, CFLAGS and
# LDFLAGS.
#
# None of these scripts end with an "exit". Exit, possibly, leads to Msys
# freezing on some versions of Windows (98).
# 
# The list order corresponds to build order. It is critical.
dependencies = [
    Dependency('SDL', ['SDL-[1-9].*'], ['SDL.dll'], """

set -e
export PATH="$PREFIX/bin:$PATH"
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

  # If this comes from the repository it has no configure script
  if [ ! -f "./configure" ]; then
    ./autogen.sh
  fi
  # Prevent libtool deadlocks (maybe).
  ./configure --disable-libtool-lock --prefix="$PREFIX" LDFLAGS="$LDFLAGS"
  
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
  # Make SDL_config_win32.h available for prebuilt and MSVC
  cp -f "$BDWD/include/SDL_config_win32.h" "$PREFIX/include/SDL"
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/SDL.dll"
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('Z', ['zlib-[1-9].*'], ['zlib1.dll'], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  cp -fp win32/Makefile.gcc .
  # Will use contributed asm code.
  cp -fp contrib/asm686/match.S .
fi

if [ x$BDCOMP == x1 ]; then
  # Build with the import library renamed, using asm code, our CFLAGS
  # and LDFLAGS.
  make IMPLIB=libz.dll.a CFLAGS="$CFLAGS" LOC="-DASMV $LDFLAGS" \
    OBJA=match.o -fMakefile.gcc
fi

if [ x$BDINST == x1 ]; then
  # Make sure everything is installed in the correct places
  make install LIBRARY_PATH="$PREFIX/lib" INCLUDE_PATH="$PREFIX/include" \
    BINARY_PATH="$PREFIX/bin" SHARED_MODE=1 IMPLIB=libz.dll.a -fMakefile.gcc
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/zlib1.dll"
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean -fMakefile.gcc
fi
"""),
    Dependency('FREETYPE', ['freetype-[2-9].*'], ['libfreetype-6.dll'], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  ./configure --prefix="$PREFIX" LDFLAGS="$LDFLAGS"
  
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' builds/unix/config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/libfreetype-6.dll"
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('FONT', ['SDL_ttf-[2-9].*'], ['SDL_ttf.dll'], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  ./configure --prefix="$PREFIX" LDFLAGS="$LDFLAGS"
  
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/SDL_ttf.dll"
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('PNG', ['l*png*[1-9][1-9.]*'], ['libpng16-16.dll'], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  ./configure --prefix="$PREFIX" CPPFLAGS="$CPPFLAGS" LDFLAGS="$LDFLAGS"
  
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/libpng16-16.dll"
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean -fMakefile.mingw prefix="$PREFIX"
fi
"""),
    Dependency('JPEG', ['jpeg-[6-9]*'], ['libjpeg-8.dll'], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  # This will only build a static library.
  ./configure --prefix="$PREFIX" LDFLAGS="$LDFLAGS"
  
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi

  cp jconfig.vc jconfig.h
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  # Only install the headers and import library, otherwise SDL_image will
  # statically link to jpeg.
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/libjpeg-8.dll"
fi

if [ x$BDCLEAN == x1 ]; then
  set +e
  make clean
fi
"""),
    Dependency('TIFF', ['tiff-[3-9].*'], ['libtiff-5.dll'], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  ./configure --disable-cxx --prefix="$PREFIX" LDFLAGS="$LDFLAGS"
  
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/libtiff-5.dll"
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
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  # If this comes from the repository it has no configure script
  if [ ! -f "./configure" ]; then
    ./autogen.sh
  fi

  # configure searches for the JPEG dll. Unfortunately it uses the wrong file
  # name. Correct this.
  mv configure configure~
  sed -e 's|jpeg\.dll|libjpeg-*.dll|' configure~ >configure
  
  # Add the destination bin directory to the library search path so
  # configure can find its precious DLL files.
  export LDFLAGS="$LDFLAGS -L$PREFIX/bin"
  
  # Add path to PNG headers
  CPPFLAGS="$CPPFLAGS `$PREFIX/bin/libpng-config --I_opts`"
  
  # Disable dynamic loading of image libraries as it uses the wrong DLL
  # search path: does not check in the same directory.
  #  --disable-libtool-lock: Prevent libtool deadlocks (maybe).
  ./configure --disable-jpg-shared --disable-png-shared --disable-tif-shared \
              --disable-libtool-lock \
              --prefix="$PREFIX" CPPFLAGS="$CPPFLAGS" LDFLAGS="$LDFLAGS"
  
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/SDL_image.dll"
fi

if [[ x$BDCLEAN == x1 && -f Makefile ]]; then
  set +e
  make clean
fi
"""),
    Dependency('SMPEG', ['smpeg-[0-9].*', 'smpeg'], ['smpeg.dll'], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  # This comes straight from SVN so has no configure script
  if [ ! -f "./configure" ]; then
    ./autogen.sh
  fi

  # Don't need the toys. Disable dynamic linking of libgcc and libstdc++
  ./configure --disable-gtk-player --disable-opengl-player \
              --prefix="$PREFIX" CFLAGS="-static-libgcc $CFLAGS"
              
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi
fi

if [ x$BDCOMP == x1 ]; then
  # Leave out undefined symbols so a dll will build.
  make CXXLD='$(CXX) -no-undefined'
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/smpeg.dll"
fi

if [[ x$BDCLEAN == x1 && -f Makefile ]]; then
  set +e
  make clean
fi
"""),
    Dependency('OGG', ['libogg-[1-9].*'], ['libogg-0.dll'], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  ./configure --prefix="$PREFIX" LDFLAGS="$LDFLAGS"
  
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/libogg-0.dll"
fi

if [[ x$BDCLEAN == x1 && -f Makefile ]]; then
  set +e
  make clean
fi
"""),
    Dependency('VORBIS',
               ['libvorbis-[1-9].*'],
               ['libvorbis-0.dll', 'libvorbisfile-3.dll'], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  ./configure --prefix="$PREFIX" LDFLAGS="$LDFLAGS" LIBS='-logg'
  
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/libvorbis-0.dll"
  strip --strip-all "$PREFIX/bin/libvorbisfile-3.dll"
fi

if [[ x$BDCLEAN == x1 && -f Makefile ]]; then
  set +e
  make clean
fi
"""),
    Dependency('FLAC', ['flac-[1-9].*'], ['libFLAC.a'], """

set -e
export PATH="$PREFIX/bin:$PATH"
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
    --disable-doxygen-docs --prefix="$PREFIX" LDFLAGS="$LDFLAGS"
  
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  cp src/libFLAC/.libs/libFLAC.a "$PREFIX/lib"
  mkdir -p "$PREFIX/include/FLAC"
  cp -f include/FLAC/*.h "$PREFIX/include/FLAC"
fi

if [[ x$BDCLEAN == x1 && -f Makefile ]]; then
  set +e
  make clean
fi
"""),
    Dependency('MIKMOD', ['libmikmod-3.*'], ['libmikmod.a'], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then

  function mikmod_getver
  {
    gawk '\
      function getnum(s)
      {
        match(s, /[0-9]+/)
        return substr(s, RSTART, RLENGTH)
      }
      /^LIBMIKMOD_MAJOR_VERSION *= *[0-9]+/ { major = getnum($0); next}
      /^LIBMIKMOD_MINOR_VERSION *= *[0-9]+/ { minor = getnum($0); next}
      /^LIBMIKMOD_MICRO_VERSION *= *[0-9]+/ { micro = getnum($0); next}
      END { printf "%s %s %s", major, minor, micro }' \
      $1
    }

  export mikmod_version=(`mikmod_getver configure.in`)
  sed -e "s~@prefix@~$PREFIX~g" \
      -e "s~@exec_prefix@~$PREFIX~g" \
      -e "s~@LIBMIKMOD_VERSION@~${mikmod_version[0]}.${mikmod_version[1]}.${mikmod_version[2]}~g" \
      -e "s~@REENTRANT@~-D_REENTRANT~g" \
      -e "s~@LIB_LDADD@~~g" \
      -e "s~@LIBRARY_LIB@~-lpthread~g" \
      libmikmod-config.in >libmikmod-config

  cat > win32/Makefile.static.mingw << THE_END
# MinGW Makefile adapted from template for use under win32
#
# libmikmod subdirectory

# Your compiler here
CC=gcc
# Compiler flags
CPPFLAGS_MIKMOD=-c -DWIN32 -DDRV_DS -DDRV_WIN -DHAVE_FCNTL_H -DHAVE_MALLOC_H -DHAVE_LIMITS_H \\$(CPPFLAGS)
COMPILE=\\$(CC) \\$(CPPFLAGS_MIKMOD) -I../include -I.. -I../win32 \\$(CFLAGS)

.SUFFIXES:
.SUFFIXES: .o .c

LIBNAME=libmikmod.a

LIBS=\\$(LIBNAME)

DRIVER_OBJ=drv_ds.o drv_win.o

OBJ=\\$(DRIVER_OBJ) \\
    drv_nos.o drv_raw.o drv_stdout.o drv_wav.o \\
    load_669.o load_amf.o load_dsm.o load_far.o load_gdm.o load_it.o  \\
    load_imf.o load_m15.o load_med.o load_mod.o load_mtm.o load_okt.o \\
    load_s3m.o load_stm.o load_stx.o load_ult.o load_uni.o load_xm.o \\
    mmalloc.o mmerror.o mmio.o \\
    mdriver.o mdreg.o mloader.o mlreg.o mlutil.o mplayer.o munitrk.o mwav.o \\
    npertab.o sloader.o virtch.o virtch2.o virtch_common.o

all:            \\$(LIBS)

clean:
\tfor f in \\$(LIBS) ; do rm -f $f; done
\trm -f *.o
\trm -f mikmod_build.h

distclean:
\trm -f ../include/mikmod.h

install:
\tcp -fp libmikmod.a "\\$(PREFIX)/lib"
\tcp -fp ../include/mikmod.h "\\$(PREFIX)/include"
\tcp -fp ../libmikmod-config "\\$(PREFIX)/bin"

\\$(LIBNAME):     \\$(OBJ)
\tar -r \\$(LIBNAME) *.o
\tranlib \\$(LIBNAME)

../include/mikmod.h ../win32/mikmod_build.h:\t../include/mikmod.h.in
\tsed -e "s~@LIBMIKMOD_MAJOR_VERSION@~${mikmod_version[0]}~" \\
\t    -e "s~@LIBMIKMOD_MINOR_VERSION@~${mikmod_version[1]}~" \\
\t    -e "s~@LIBMIKMOD_MICRO_VERSION@~${mikmod_version[2]}~" \\
\t    -e "s~@DOES_NOT_HAVE_SIGNED@~~" \\
\t    ../include/mikmod.h.in >../win32/mikmod_build.h
\tcp -f ../win32/mikmod_build.h ../include/mikmod.h

drv_ds.o:       ../drivers/drv_ds.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../drivers/drv_ds.c
drv_nos.o:      ../drivers/drv_nos.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../drivers/drv_nos.c
drv_raw.o:      ../drivers/drv_raw.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../drivers/drv_raw.c
drv_stdout.o:   ../drivers/drv_stdout.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../drivers/drv_stdout.c
drv_wav.o:      ../drivers/drv_wav.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../drivers/drv_wav.c
drv_win.o:       ../drivers/drv_win.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../drivers/drv_win.c
load_669.o:     ../loaders/load_669.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_669.c
load_amf.o:     ../loaders/load_amf.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_amf.c
load_dsm.o:     ../loaders/load_dsm.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_dsm.c
load_far.o:     ../loaders/load_far.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_far.c
load_gdm.o:     ../loaders/load_gdm.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_gdm.c
load_it.o:      ../loaders/load_it.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_it.c
load_imf.o:     ../loaders/load_imf.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_imf.c
load_m15.o:     ../loaders/load_m15.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_m15.c
load_med.o:     ../loaders/load_med.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_med.c
load_mod.o:     ../loaders/load_mod.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_mod.c
load_mtm.o:     ../loaders/load_mtm.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_mtm.c
load_okt.o:     ../loaders/load_okt.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_okt.c
load_s3m.o:     ../loaders/load_s3m.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_s3m.c
load_stm.o:     ../loaders/load_stm.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_stm.c
load_stx.o:     ../loaders/load_stx.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_stx.c
load_ult.o:     ../loaders/load_ult.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_ult.c
load_uni.o:     ../loaders/load_uni.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_uni.c
load_xm.o:      ../loaders/load_xm.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../loaders/load_xm.c
mmalloc.o:      ../mmio/mmalloc.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../mmio/mmalloc.c
mmerror.o:      ../mmio/mmerror.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../mmio/mmerror.c
mmio.o:         ../mmio/mmio.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../mmio/mmio.c
mdriver.o:      ../playercode/mdriver.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/mdriver.c
mdreg.o:        ../playercode/mdreg.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/mdreg.c
mloader.o:      ../playercode/mloader.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/mloader.c
mlreg.o:        ../playercode/mlreg.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/mlreg.c
mlutil.o:       ../playercode/mlutil.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/mlutil.c
mplayer.o:      ../playercode/mplayer.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/mplayer.c
munitrk.o:      ../playercode/munitrk.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/munitrk.c
mwav.o:         ../playercode/mwav.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/mwav.c
npertab.o:      ../playercode/npertab.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/npertab.c
sloader.o:      ../playercode/sloader.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/sloader.c
virtch.o:       ../playercode/virtch.c ../playercode/virtch_common.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/virtch.c
virtch2.o:      ../playercode/virtch2.c ../playercode/virtch_common.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/virtch2.c
virtch_common.o:        ../playercode/virtch_common.c \\
\t                ../win32/mikmod_build.h ../include/mikmod_internals.h
\t\\$(COMPILE) -o \\$@ ../playercode/virtch_common.c
THE_END

    unset -v mikmod_version
fi

if [ x$BDCOMP == x1 ]; then
  cd win32
  make LDFLAGS="$LDFLAGS" -fMakefile.static.mingw
  cd ..
fi

if [ x$BDINST == x1 ]; then
  cd win32
  make install PREFIX="$PREFIX" -fMakefile.static.mingw
  cd ..
fi

if [[ x$BDCLEAN == x1 && -f Makefile ]]; then
  cd win32
  set +e
  make clean -fMakefile.static.mingw
  rm -f Makefile.static.mingw
  cd ..
  rm -f libmikmod-config
fi

"""),
    Dependency('MIXER', ['SDL_mixer-[1-9].*'], ['SDL_mixer.dll'], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

mikmod_dependencies='-ldsound'
flac_dependencies='-lWs2_32'

if [ x$BDCONF == x1 ]; then
  # If this came from SVN then need a configure script.
  if [ ! -f "./configure" ]; then
    ./autogen.sh
  fi

  # No dynamic loading of dependent libraries. Use LIBS so FLAC test
  # builds (unfortunately LIBS is not passed on to Makefile).
  export LIBS="$mikmod_dependencies $flac_dependencies"
  ./configure --disable-music-ogg-shared --disable-music-mp3-shared \
    --disable-music-mod-shared --disable-music-flac-shared \
    --disable-libtool-lock --prefix="$PREFIX" LDFLAGS="$LDFLAGS"
  
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi
  
  # ./configure messes up on its Makefile generation, putting some rules
  # on the same line as their targets, and placing multiple targets in one
  # line. Break them up. Also add the required FLAC and mikmod linkage flags.
  mv -f Makefile Makefile~
  sed -e 's~\\(-c $< -o $@\\) \\($(objects)\\)~\\1\\\n\\2~g' \
      -e 's~\\(\\.c\\)\\(\t$(LIBTOOL)\\)~\\1\\\n\\2~g' \
      -e 's~\\(: \\./version.rc\\)\\(\t$(WINDRES)\\)~\\1\\\n\\2~' \
      -e "s~\\(-lFLAC\\)~\\1 $flac_dependencies~" \
      -e "s~\\(-lmikmod\\)~\\1 $mikmod_dependencies~" \
      Makefile~ >Makefile
fi

if [ x$BDCOMP == x1 ]; then
  make
fi

if [ x$BDINST == x1 ]; then
  make install
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/SDL_mixer.dll"
fi

if [[ x$BDCLEAN == x1 && -f Makefile ]]; then
  set +e
  make clean
fi
"""),
    Dependency('PORTMIDI', ['portmidi', 'portmidi-[1-9].*'], ['portmidi.dll'], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  # Fix up some g++ 4.5.0 issues in the source code.
  source_file=pm_common/portmidi.c
  if [ ! -f "$source_file~" ]; then
     mv "$source_file" "$source_file~"
     sed \
'420,+7s/return !Pm_QueueEmpty(midi->queue);/\
return Pm_QueueEmpty(midi->queue) ? pmNoData : pmGotData;/' \
         "$source_file~" >"$source_file"
  fi
  source_file=pm_win/pmwin.c
if [ ! -f "$source_file~" ]; then
     mv "$source_file" "$source_file~"
     sed \
-e '20,+6s/^\\(#include <windows.h>\\)/#include <ctype.h>\\\n\\1/' \
-e '91,+7s/if (RegQueryValueEx(hkey, key, NULL, &dwType, pattern, &pattern_max) !=/\
if (RegQueryValueEx(hkey, key, NULL, \\&dwType, (BYTE *) pattern, (DWORD *) \\&pattern_max) !=/' \
         "$source_file~" >"$source_file"
  fi
  source_file=pm_win/pmwinmm.c
  if [ ! -f "$source_file~" ]; then
     mv "$source_file" "$source_file~"
     sed \
-e '207,+7s/midi_out_caps = pm_alloc( sizeof(MIDIOUTCAPS) \\* midi_num_outputs );/\
midi_out_caps = (MIDIOUTCAPS *) pm_alloc( sizeof(MIDIOUTCAPS) * midi_num_outputs );/' \
-e '531,+10s/return pm_hosterror;/return pmInsufficientMemory;/' \
-e '531,+10s/return pm_hosterror;/return pm_hosterror ? pmInsufficientMemory : pmNoError;/' \
-e '626,+7s/return m->error;/return m->error == MMSYSERR_NOERROR ? pmNoError : pmHostError;/' \
-e '1206,+7s/midi->fill_offset_ptr = &(hdr->dwBytesRecorded);/\
midi->fill_offset_ptr = (uint32_t *) \\&(hdr->dwBytesRecorded);/' \
-e '1422,+7s/PmInternal \\* midi = descriptors\\[i\\]\\.internalDescriptor;/\
PmInternal * midi = (PmInternal *) descriptors[i].internalDescriptor;/' \
         "$source_file~" >"$source_file"
  fi

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
\tg++ -shared -static-libgcc $(LDFLAGS) -def $(def) $(pmlib) $(LIBS) -o \\$@
\tdlltool -D $(pmdll) -d $(def) -l $(pmimplib)
\tranlib $(pmimplib)

.PHONY : install

install : $(pmdll)
\tcp -f --target-directory=$(PREFIX)/bin $<
\tcp -f --target-directory=$(PREFIX)/lib $(pmlib)
\tcp -f --target-directory=$(PREFIX)/lib $(pmimplib)
\tcp -f --target-directory=$(PREFIX)/include $(reqhdr)

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
  make LDFLAGS="$LDFLAGS"
fi

if [ x$BDINST == x1 ]; then
  make install PREFIX="$PREFIX"
fi

if [ x$BDSTRIP == x1 ]; then
  strip --strip-all "$PREFIX/bin/portmidi.dll"
fi

if [[ x$BDCLEAN == x1 && -f Makefile ]]; then
  set +e
  make clean
  rm -f GNUmakefile portmidi.def
fi
"""),
    Dependency('FFMPEG', ['ffmpeg'],
    ['avformat-52.dll', 'swscale-0.dll',
     'avcodec-52.dll', 'avutil-50.dll' ], """

set -e
export PATH="$PREFIX/bin:$PATH"
cd "$BDWD"

if [ x$BDCONF == x1 ]; then
  # Don't want the pthreads dll, which links to msvcrt.dll.
  ./configure --enable-shared --enable-memalign-hack \
              --disable-pthreads --prefix="$PREFIX" \
              --enable-runtime-cpudetect
              
  # check for MSYS permission errors
  if [ x"`grep 'Permission denied' config.log`" != x ]; then
      echo '**** MSYS problems; build aborted.'
      exit 1
  fi

  # Fix incompatibilities between ffmpeg and MinGW notions of the C99 standard.
  mv config.mak config.mak~
  sed -e "s~\\\\(-std=\\\\)c99~\\\\1gnu99~g" \
      config.mak~ >config.mak
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
  strip --strip-all "$PREFIX/bin/avformat-52.dll"
  strip --strip-all "$PREFIX/bin/swscale-0.dll"
  strip --strip-all "$PREFIX/bin/avcodec-52.dll"
  strip --strip-all "$PREFIX/bin/avutil-50.dll"
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


msys_prep = Preparation('destintation directory', """

# Ensure the destination directory and its subdirectories exist.
mkdir -p "$PREFIX/lib"
mkdir -p "$PREFIX/include"
mkdir -p "$PREFIX/bin"
mkdir -p "$PREFIX/doc"
mkdir -p "$PREFIX/man"
mkdir -p "$PREFIX/share"
""")
    
msvcr71_prep = Preparation('msvcr71.dll linkage', r"""

set -e

#
#   msvcr71.dll support
#
if [ ! -f "$BDMSVCR71/libmoldname.a" ]; then
  echo "Making directory $BDMSVCR71 for msvcr71.dll linking."
  mkdir -p "$BDMSVCR71"
  cp -fp /mingw/lib/libmoldname71.a "$BDMSVCR71/libmoldname.a"
  cp -fp /mingw/lib/libmoldname71d.a "$BDMSVCR71/libmoldnamed.a"
  cp -fp /mingw/lib/libmsvcr71.a "$BDMSVCR71/libmsvcrt.a"
  cp -fp /mingw/lib/libmsvcr71d.a "$BDMSVCR71/libmsvcrtd.a"
fi
""")

msvcr90_prep = Preparation('msvcr90.dll linkage', r"""

set -e

#
#   msvcr90.dll support
#
if [ ! -f "$BDMSVCR90/libmoldnamed.dll.a" ]; then
  echo Adding libraries to $BDMSVCR90
  mkdir -p "$BDMSVCR90"
  OBJS='isascii.o iscsym.o iscsymf.o toascii.o
        strcasecmp.o strncasecmp.o wcscmpi.o'
  if [ ! -d /tmp/build_deps ]; then mkdir /tmp/build_deps; fi
  cd /tmp/build_deps

  # These definitions were generated with pexports on msvcr90.dll.
  # The C++ stuff at the beginning was removed. _onexit and atexit made
  # data entries.
  cat > msvcr90.def << 'THE_END'
EXPORTS
_CIacos
_CIasin
_CIatan
_CIatan2
_CIcos
_CIcosh
_CIexp
_CIfmod
_CIlog
_CIlog10
_CIpow
_CIsin
_CIsinh
_CIsqrt
_CItan
_CItanh
_CRT_RTC_INIT
_CRT_RTC_INITW
_CreateFrameInfo
_CxxThrowException
_EH_prolog
_FindAndUnlinkFrame
_Getdays
_Getmonths
_Gettnames
_HUGE DATA
_IsExceptionObjectToBeDestroyed
_NLG_Dispatch2
_NLG_Return
_NLG_Return2
_Strftime
_XcptFilter
__AdjustPointer
__BuildCatchObject
__BuildCatchObjectHelper
__CppXcptFilter
__CxxCallUnwindDelDtor
__CxxCallUnwindDtor
__CxxCallUnwindStdDelDtor
__CxxCallUnwindVecDtor
__CxxDetectRethrow
__CxxExceptionFilter
__CxxFrameHandler
__CxxFrameHandler2
__CxxFrameHandler3
__CxxLongjmpUnwind
__CxxQueryExceptionSize
__CxxRegisterExceptionObject
__CxxUnregisterExceptionObject
__DestructExceptionObject
__FrameUnwindFilter
__RTCastToVoid
__RTDynamicCast
__RTtypeid
__STRINGTOLD
__STRINGTOLD_L
__TypeMatch
___fls_getvalue@4
___fls_setvalue@8
___lc_codepage_func
___lc_collate_cp_func
___lc_handle_func
___mb_cur_max_func
___mb_cur_max_l_func
___setlc_active_func
___unguarded_readlc_active_add_func
__argc DATA
__argv DATA
__badioinfo DATA
__clean_type_info_names_internal
__control87_2
__create_locale
__crtCompareStringA
__crtCompareStringW
__crtGetLocaleInfoW
__crtGetStringTypeW
__crtLCMapStringA
__crtLCMapStringW
__daylight
__dllonexit
__doserrno
__dstbias
__fpecode
__free_locale
__get_app_type
__get_current_locale
__get_flsindex
__get_tlsindex
__getmainargs
__initenv DATA
__iob_func
__isascii
__iscsym
__iscsymf
__iswcsym
__iswcsymf
__lc_clike DATA
__lc_codepage DATA
__lc_collate_cp DATA
__lc_handle DATA
__lconv DATA
__lconv_init
__libm_sse2_acos
__libm_sse2_acosf
__libm_sse2_asin
__libm_sse2_asinf
__libm_sse2_atan
__libm_sse2_atan2
__libm_sse2_atanf
__libm_sse2_cos
__libm_sse2_cosf
__libm_sse2_exp
__libm_sse2_expf
__libm_sse2_log
__libm_sse2_log10
__libm_sse2_log10f
__libm_sse2_logf
__libm_sse2_pow
__libm_sse2_powf
__libm_sse2_sin
__libm_sse2_sinf
__libm_sse2_tan
__libm_sse2_tanf
__mb_cur_max DATA
__p___argc
__p___argv
__p___initenv
__p___mb_cur_max
__p___wargv
__p___winitenv
__p__acmdln
__p__amblksiz
__p__commode
__p__daylight
__p__dstbias
__p__environ
__p__fmode
__p__iob
__p__mbcasemap
__p__mbctype
__p__pctype
__p__pgmptr
__p__pwctype
__p__timezone
__p__tzname
__p__wcmdln
__p__wenviron
__p__wpgmptr
__pctype_func
__pioinfo DATA
__pwctype_func
__pxcptinfoptrs
__report_gsfailure
__set_app_type
__set_flsgetvalue
__setlc_active DATA
__setusermatherr
__strncnt
__swprintf_l
__sys_errlist
__sys_nerr
__threadhandle
__threadid
__timezone
__toascii
__tzname
__unDName
__unDNameEx
__unDNameHelper
__uncaught_exception
__unguarded_readlc_active DATA
__vswprintf_l
__wargv DATA
__wcserror
__wcserror_s
__wcsncnt
__wgetmainargs
__winitenv DATA
_abnormal_termination
_abs64
_access
_access_s
_acmdln DATA
_adj_fdiv_m16i
_adj_fdiv_m32
_adj_fdiv_m32i
_adj_fdiv_m64
_adj_fdiv_r
_adj_fdivr_m16i
_adj_fdivr_m32
_adj_fdivr_m32i
_adj_fdivr_m64
_adj_fpatan
_adj_fprem
_adj_fprem1
_adj_fptan
_adjust_fdiv DATA
_aexit_rtn DATA
_aligned_free
_aligned_malloc
_aligned_msize
_aligned_offset_malloc
_aligned_offset_realloc
_aligned_offset_recalloc
_aligned_realloc
_aligned_recalloc
_amsg_exit
_assert
_atodbl
_atodbl_l
_atof_l
_atoflt
_atoflt_l
_atoi64
_atoi64_l
_atoi_l
_atol_l
_atoldbl
_atoldbl_l
_beep
_beginthread
_beginthreadex
_byteswap_uint64
_byteswap_ulong
_byteswap_ushort
_c_exit
_cabs
_callnewh
_calloc_crt
_cexit
_cgets
_cgets_s
_cgetws
_cgetws_s
_chdir
_chdrive
_chgsign
_chkesp
_chmod
_chsize
_chsize_s
_clearfp
_close
_commit
_commode DATA
_configthreadlocale
_control87
_controlfp
_controlfp_s
_copysign
_cprintf
_cprintf_l
_cprintf_p
_cprintf_p_l
_cprintf_s
_cprintf_s_l
_cputs
_cputws
_creat
_create_locale
_crt_debugger_hook
_cscanf
_cscanf_l
_cscanf_s
_cscanf_s_l
_ctime32
_ctime32_s
_ctime64
_ctime64_s
_cwait
_cwprintf
_cwprintf_l
_cwprintf_p
_cwprintf_p_l
_cwprintf_s
_cwprintf_s_l
_cwscanf
_cwscanf_l
_cwscanf_s
_cwscanf_s_l
_daylight DATA
_decode_pointer
_difftime32
_difftime64
_dosmaperr
_dstbias DATA
_dup
_dup2
_dupenv_s
_ecvt
_ecvt_s
_encode_pointer
_encoded_null
_endthread
_endthreadex
_environ DATA
_eof
_errno
_except_handler2
_except_handler3
_except_handler4_common
_execl
_execle
_execlp
_execlpe
_execv
_execve
_execvp
_execvpe
_exit
_expand
_fclose_nolock
_fcloseall
_fcvt
_fcvt_s
_fdopen
_fflush_nolock
_fgetchar
_fgetwc_nolock
_fgetwchar
_filbuf
_filelength
_filelengthi64
_fileno
_findclose
_findfirst32
_findfirst32i64
_findfirst64
_findfirst64i32
_findnext32
_findnext32i64
_findnext64
_findnext64i32
_finite
_flsbuf
_flushall
_fmode DATA
_fpclass
_fpieee_flt
_fpreset
_fprintf_l
_fprintf_p
_fprintf_p_l
_fprintf_s_l
_fputchar
_fputwc_nolock
_fputwchar
_fread_nolock
_fread_nolock_s
_free_locale
_freea
_freea_s
_freefls
_fscanf_l
_fscanf_s_l
_fseek_nolock
_fseeki64
_fseeki64_nolock
_fsopen
_fstat32
_fstat32i64
_fstat64
_fstat64i32
_ftell_nolock
_ftelli64
_ftelli64_nolock
_ftime32
_ftime32_s
_ftime64
_ftime64_s
_ftol
_fullpath
_futime32
_futime64
_fwprintf_l
_fwprintf_p
_fwprintf_p_l
_fwprintf_s_l
_fwrite_nolock
_fwscanf_l
_fwscanf_s_l
_gcvt
_gcvt_s
_get_amblksiz
_get_current_locale
_get_daylight
_get_doserrno
_get_dstbias
_get_errno
_get_fmode
_get_heap_handle
_get_invalid_parameter_handler
_get_osfhandle
_get_output_format
_get_pgmptr
_get_printf_count_output
_get_purecall_handler
_get_sbh_threshold
_get_terminate
_get_timezone
_get_tzname
_get_unexpected
_get_wpgmptr
_getc_nolock
_getch
_getch_nolock
_getche
_getche_nolock
_getcwd
_getdcwd
_getdcwd_nolock
_getdiskfree
_getdllprocaddr
_getdrive
_getdrives
_getmaxstdio
_getmbcp
_getpid
_getptd
_getsystime
_getw
_getwch
_getwch_nolock
_getwche
_getwche_nolock
_getws
_getws_s
_global_unwind2
_gmtime32
_gmtime32_s
_gmtime64
_gmtime64_s
_heapadd
_heapchk
_heapmin
_heapset
_heapused
_heapwalk
_hypot
_hypotf
_i64toa
_i64toa_s
_i64tow
_i64tow_s
_initptd
_initterm
_initterm_e
_inp
_inpd
_inpw
_invalid_parameter
_invalid_parameter_noinfo
_invoke_watson
_iob DATA
_isalnum_l
_isalpha_l
_isatty
_iscntrl_l
_isctype
_isctype_l
_isdigit_l
_isgraph_l
_isleadbyte_l
_islower_l
_ismbbalnum
_ismbbalnum_l
_ismbbalpha
_ismbbalpha_l
_ismbbgraph
_ismbbgraph_l
_ismbbkalnum
_ismbbkalnum_l
_ismbbkana
_ismbbkana_l
_ismbbkprint
_ismbbkprint_l
_ismbbkpunct
_ismbbkpunct_l
_ismbblead
_ismbblead_l
_ismbbprint
_ismbbprint_l
_ismbbpunct
_ismbbpunct_l
_ismbbtrail
_ismbbtrail_l
_ismbcalnum
_ismbcalnum_l
_ismbcalpha
_ismbcalpha_l
_ismbcdigit
_ismbcdigit_l
_ismbcgraph
_ismbcgraph_l
_ismbchira
_ismbchira_l
_ismbckata
_ismbckata_l
_ismbcl0
_ismbcl0_l
_ismbcl1
_ismbcl1_l
_ismbcl2
_ismbcl2_l
_ismbclegal
_ismbclegal_l
_ismbclower
_ismbclower_l
_ismbcprint
_ismbcprint_l
_ismbcpunct
_ismbcpunct_l
_ismbcspace
_ismbcspace_l
_ismbcsymbol
_ismbcsymbol_l
_ismbcupper
_ismbcupper_l
_ismbslead
_ismbslead_l
_ismbstrail
_ismbstrail_l
_isnan
_isprint_l
_ispunct_l
_isspace_l
_isupper_l
_iswalnum_l
_iswalpha_l
_iswcntrl_l
_iswcsym_l
_iswcsymf_l
_iswctype_l
_iswdigit_l
_iswgraph_l
_iswlower_l
_iswprint_l
_iswpunct_l
_iswspace_l
_iswupper_l
_iswxdigit_l
_isxdigit_l
_itoa
_itoa_s
_itow
_itow_s
_j0
_j1
_jn
_kbhit
_lfind
_lfind_s
_loaddll
_local_unwind2
_local_unwind4
_localtime32
_localtime32_s
_localtime64
_localtime64_s
_lock
_lock_file
_locking
_logb
_longjmpex
_lrotl
_lrotr
_lsearch
_lsearch_s
_lseek
_lseeki64
_ltoa
_ltoa_s
_ltow
_ltow_s
_makepath
_makepath_s
_malloc_crt
_mbbtombc
_mbbtombc_l
_mbbtype
_mbbtype_l
_mbcasemap DATA
_mbccpy
_mbccpy_l
_mbccpy_s
_mbccpy_s_l
_mbcjistojms
_mbcjistojms_l
_mbcjmstojis
_mbcjmstojis_l
_mbclen
_mbclen_l
_mbctohira
_mbctohira_l
_mbctokata
_mbctokata_l
_mbctolower
_mbctolower_l
_mbctombb
_mbctombb_l
_mbctoupper
_mbctoupper_l
_mbctype DATA
_mblen_l
_mbsbtype
_mbsbtype_l
_mbscat_s
_mbscat_s_l
_mbschr
_mbschr_l
_mbscmp
_mbscmp_l
_mbscoll
_mbscoll_l
_mbscpy_s
_mbscpy_s_l
_mbscspn
_mbscspn_l
_mbsdec
_mbsdec_l
_mbsicmp
_mbsicmp_l
_mbsicoll
_mbsicoll_l
_mbsinc
_mbsinc_l
_mbslen
_mbslen_l
_mbslwr
_mbslwr_l
_mbslwr_s
_mbslwr_s_l
_mbsnbcat
_mbsnbcat_l
_mbsnbcat_s
_mbsnbcat_s_l
_mbsnbcmp
_mbsnbcmp_l
_mbsnbcnt
_mbsnbcnt_l
_mbsnbcoll
_mbsnbcoll_l
_mbsnbcpy
_mbsnbcpy_l
_mbsnbcpy_s
_mbsnbcpy_s_l
_mbsnbicmp
_mbsnbicmp_l
_mbsnbicoll
_mbsnbicoll_l
_mbsnbset
_mbsnbset_l
_mbsnbset_s
_mbsnbset_s_l
_mbsncat
_mbsncat_l
_mbsncat_s
_mbsncat_s_l
_mbsnccnt
_mbsnccnt_l
_mbsncmp
_mbsncmp_l
_mbsncoll
_mbsncoll_l
_mbsncpy
_mbsncpy_l
_mbsncpy_s
_mbsncpy_s_l
_mbsnextc
_mbsnextc_l
_mbsnicmp
_mbsnicmp_l
_mbsnicoll
_mbsnicoll_l
_mbsninc
_mbsninc_l
_mbsnlen
_mbsnlen_l
_mbsnset
_mbsnset_l
_mbsnset_s
_mbsnset_s_l
_mbspbrk
_mbspbrk_l
_mbsrchr
_mbsrchr_l
_mbsrev
_mbsrev_l
_mbsset
_mbsset_l
_mbsset_s
_mbsset_s_l
_mbsspn
_mbsspn_l
_mbsspnp
_mbsspnp_l
_mbsstr
_mbsstr_l
_mbstok
_mbstok_l
_mbstok_s
_mbstok_s_l
_mbstowcs_l
_mbstowcs_s_l
_mbstrlen
_mbstrlen_l
_mbstrnlen
_mbstrnlen_l
_mbsupr
_mbsupr_l
_mbsupr_s
_mbsupr_s_l
_mbtowc_l
_memccpy
_memicmp
_memicmp_l
_mkdir
_mkgmtime32
_mkgmtime64
_mktemp
_mktemp_s
_mktime32
_mktime64
_msize
_nextafter
_onexit DATA
_open
_open_osfhandle
_outp
_outpd
_outpw
_pclose
_pctype DATA
_pgmptr DATA
_pipe
_popen
_printf_l
_printf_p
_printf_p_l
_printf_s_l
_purecall
_putch
_putch_nolock
_putenv
_putenv_s
_putw
_putwch
_putwch_nolock
_putws
_pwctype DATA
_read
_realloc_crt
_recalloc
_recalloc_crt
_resetstkoflw
_rmdir
_rmtmp
_rotl
_rotl64
_rotr
_rotr64
_safe_fdiv
_safe_fdivr
_safe_fprem
_safe_fprem1
_scalb
_scanf_l
_scanf_s_l
_scprintf
_scprintf_l
_scprintf_p
_scprintf_p_l
_scwprintf
_scwprintf_l
_scwprintf_p
_scwprintf_p_l
_searchenv
_searchenv_s
_seh_longjmp_unwind
_seh_longjmp_unwind4
_set_SSE2_enable
_set_abort_behavior
_set_amblksiz
_set_controlfp
_set_doserrno
_set_errno
_set_error_mode
_set_fmode
_set_invalid_parameter_handler
_set_malloc_crt_max_wait
_set_output_format
_set_printf_count_output
_set_purecall_handler
_set_sbh_threshold
_seterrormode
_setjmp
_setjmp3
_setmaxstdio
_setmbcp
_setmode
_setsystime
_sleep
_snprintf
_snprintf_c
_snprintf_c_l
_snprintf_l
_snprintf_s
_snprintf_s_l
_snscanf
_snscanf_l
_snscanf_s
_snscanf_s_l
_snwprintf
_snwprintf_l
_snwprintf_s
_snwprintf_s_l
_snwscanf
_snwscanf_l
_snwscanf_s
_snwscanf_s_l
_sopen
_sopen_s
_spawnl
_spawnle
_spawnlp
_spawnlpe
_spawnv
_spawnve
_spawnvp
_spawnvpe
_splitpath
_splitpath_s
_sprintf_l
_sprintf_p
_sprintf_p_l
_sprintf_s_l
_sscanf_l
_sscanf_s_l
_stat32
_stat32i64
_stat64
_stat64i32
_statusfp
_statusfp2
_strcoll_l
_strdate
_strdate_s
_strdup
_strerror
_strerror_s
_strftime_l
_stricmp
_stricmp_l
_stricoll
_stricoll_l
_strlwr
_strlwr_l
_strlwr_s
_strlwr_s_l
_strncoll
_strncoll_l
_strnicmp
_strnicmp_l
_strnicoll
_strnicoll_l
_strnset
_strnset_s
_strrev
_strset
_strset_s
_strtime
_strtime_s
_strtod_l
_strtoi64
_strtoi64_l
_strtol_l
_strtoui64
_strtoui64_l
_strtoul_l
_strupr
_strupr_l
_strupr_s
_strupr_s_l
_strxfrm_l
_swab
_swprintf
_swprintf_c
_swprintf_c_l
_swprintf_p
_swprintf_p_l
_swprintf_s_l
_swscanf_l
_swscanf_s_l
_sys_errlist DATA
_sys_nerr DATA
_tell
_telli64
_tempnam
_time32
_time64
_timezone DATA
_tolower
_tolower_l
_toupper
_toupper_l
_towlower_l
_towupper_l
_tzname DATA
_tzset
_ui64toa
_ui64toa_s
_ui64tow
_ui64tow_s
_ultoa
_ultoa_s
_ultow
_ultow_s
_umask
_umask_s
_ungetc_nolock
_ungetch
_ungetch_nolock
_ungetwc_nolock
_ungetwch
_ungetwch_nolock
_unlink
_unloaddll
_unlock
_unlock_file
_utime32
_utime64
_vcprintf
_vcprintf_l
_vcprintf_p
_vcprintf_p_l
_vcprintf_s
_vcprintf_s_l
_vcwprintf
_vcwprintf_l
_vcwprintf_p
_vcwprintf_p_l
_vcwprintf_s
_vcwprintf_s_l
_vfprintf_l
_vfprintf_p
_vfprintf_p_l
_vfprintf_s_l
_vfwprintf_l
_vfwprintf_p
_vfwprintf_p_l
_vfwprintf_s_l
_vprintf_l
_vprintf_p
_vprintf_p_l
_vprintf_s_l
_vscprintf
_vscprintf_l
_vscprintf_p
_vscprintf_p_l
_vscwprintf
_vscwprintf_l
_vscwprintf_p
_vscwprintf_p_l
_vsnprintf
_vsnprintf_c
_vsnprintf_c_l
_vsnprintf_l
_vsnprintf_s
_vsnprintf_s_l
_vsnwprintf
_vsnwprintf_l
_vsnwprintf_s
_vsnwprintf_s_l
_vsprintf_l
_vsprintf_p
_vsprintf_p_l
_vsprintf_s_l
_vswprintf
_vswprintf_c
_vswprintf_c_l
_vswprintf_l
_vswprintf_p
_vswprintf_p_l
_vswprintf_s_l
_vwprintf_l
_vwprintf_p
_vwprintf_p_l
_vwprintf_s_l
_waccess
_waccess_s
_wasctime
_wasctime_s
_wassert
_wchdir
_wchmod
_wcmdln DATA
_wcreat
_wcscoll_l
_wcsdup
_wcserror
_wcserror_s
_wcsftime_l
_wcsicmp
_wcsicmp_l
_wcsicoll
_wcsicoll_l
_wcslwr
_wcslwr_l
_wcslwr_s
_wcslwr_s_l
_wcsncoll
_wcsncoll_l
_wcsnicmp
_wcsnicmp_l
_wcsnicoll
_wcsnicoll_l
_wcsnset
_wcsnset_s
_wcsrev
_wcsset
_wcsset_s
_wcstod_l
_wcstoi64
_wcstoi64_l
_wcstol_l
_wcstombs_l
_wcstombs_s_l
_wcstoui64
_wcstoui64_l
_wcstoul_l
_wcsupr
_wcsupr_l
_wcsupr_s
_wcsupr_s_l
_wcsxfrm_l
_wctime32
_wctime32_s
_wctime64
_wctime64_s
_wctomb_l
_wctomb_s_l
_wctype
_wdupenv_s
_wenviron DATA
_wexecl
_wexecle
_wexeclp
_wexeclpe
_wexecv
_wexecve
_wexecvp
_wexecvpe
_wfdopen
_wfindfirst32
_wfindfirst32i64
_wfindfirst64
_wfindfirst64i32
_wfindnext32
_wfindnext32i64
_wfindnext64
_wfindnext64i32
_wfopen
_wfopen_s
_wfreopen
_wfreopen_s
_wfsopen
_wfullpath
_wgetcwd
_wgetdcwd
_wgetdcwd_nolock
_wgetenv
_wgetenv_s
_wmakepath
_wmakepath_s
_wmkdir
_wmktemp
_wmktemp_s
_wopen
_wperror
_wpgmptr DATA
_wpopen
_wprintf_l
_wprintf_p
_wprintf_p_l
_wprintf_s_l
_wputenv
_wputenv_s
_wremove
_wrename
_write
_wrmdir
_wscanf_l
_wscanf_s_l
_wsearchenv
_wsearchenv_s
_wsetlocale
_wsopen
_wsopen_s
_wspawnl
_wspawnle
_wspawnlp
_wspawnlpe
_wspawnv
_wspawnve
_wspawnvp
_wspawnvpe
_wsplitpath
_wsplitpath_s
_wstat32
_wstat32i64
_wstat64
_wstat64i32
_wstrdate
_wstrdate_s
_wstrtime
_wstrtime_s
_wsystem
_wtempnam
_wtmpnam
_wtmpnam_s
_wtof
_wtof_l
_wtoi
_wtoi64
_wtoi64_l
_wtoi_l
_wtol
_wtol_l
_wunlink
_wutime32
_wutime64
_y0
_y1
_yn
abort
abs
acos
asctime
asctime_s
asin
atan
atan2
atexit DATA
atof
atoi
atol
bsearch
bsearch_s
btowc
calloc
ceil
clearerr
clearerr_s
clock
cos
cosh
div
exit
exp
fabs
fclose
feof
ferror
fflush
fgetc
fgetpos
fgets
fgetwc
fgetws
floor
fmod
fopen
fopen_s
fprintf
fprintf_s
fputc
fputs
fputwc
fputws
fread
fread_s
free
freopen
freopen_s
frexp
fscanf
fscanf_s
fseek
fsetpos
ftell
fwprintf
fwprintf_s
fwrite
fwscanf
fwscanf_s
getc
getchar
getenv
getenv_s
gets
gets_s
getwc
getwchar
is_wctype
isalnum
isalpha
iscntrl
isdigit
isgraph
isleadbyte
islower
isprint
ispunct
isspace
isupper
iswalnum
iswalpha
iswascii
iswcntrl
iswctype
iswdigit
iswgraph
iswlower
iswprint
iswpunct
iswspace
iswupper
iswxdigit
isxdigit
labs
ldexp
ldiv
localeconv
log
log10
longjmp
malloc
mblen
mbrlen
mbrtowc
mbsrtowcs
mbsrtowcs_s
mbstowcs
mbstowcs_s
mbtowc
memchr
memcmp
memcpy
memcpy_s
memmove
memmove_s
memset
modf
perror
pow
printf
printf_s
putc
putchar
puts
putwc
putwchar
qsort
qsort_s
raise
rand
rand_s
realloc
remove
rename
rewind
scanf
scanf_s
setbuf
setlocale
setvbuf
signal
sin
sinh
sprintf
sprintf_s
sqrt
srand
sscanf
sscanf_s
strcat
strcat_s
strchr
strcmp
strcoll
strcpy
strcpy_s
strcspn
strerror
strerror_s
strftime
strlen
strncat
strncat_s
strncmp
strncpy
strncpy_s
strnlen
strpbrk
strrchr
strspn
strstr
strtod
strtok
strtok_s
strtol
strtoul
strxfrm
swprintf_s
swscanf
swscanf_s
system
tan
tanh
tmpfile
tmpfile_s
tmpnam
tmpnam_s
tolower
toupper
towlower
towupper
ungetc
ungetwc
vfprintf
vfprintf_s
vfwprintf
vfwprintf_s
vprintf
vprintf_s
vsprintf
vsprintf_s
vswprintf_s
vwprintf
vwprintf_s
wcrtomb
wcrtomb_s
wcscat
wcscat_s
wcschr
wcscmp
wcscoll
wcscpy
wcscpy_s
wcscspn
wcsftime
wcslen
wcsncat
wcsncat_s
wcsncmp
wcsncpy
wcsncpy_s
wcsnlen
wcspbrk
wcsrchr
wcsrtombs
wcsrtombs_s
wcsspn
wcsstr
wcstod
wcstok
wcstok_s
wcstol
wcstombs
wcstombs_s
wcstoul
wcsxfrm
wctob
wctomb
wctomb_s
wprintf
wprintf_s
wscanf
wscanf_s
THE_END

  # Provide the manifest resource
  cat > manifest.xml << 'THE_END'
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.VC90.CRT" version="9.0.21022.8" processorArchitecture="x86" publicKeyToken="1fc8b3b9a1e18e3b"></assemblyIdentity>
    </dependentAssembly>
  </dependency>
</assembly>
THE_END

  cat > manifest.rc << 'THE_END'
#include <winuser.h>
1 RT_MANIFEST manifest.xml
THE_END

  windres -o "$BDRESOURCES" manifest.rc
  
  # Provide the gmtime stub required by PNG.
  cat > gmtime.c << 'THE_END'
/* Stub function for gmtime.
 * This is an inline function in Visual C 2008 so is missing from msvcr90.dll
 */
#include <time.h>

struct tm* _gmtime32(const time_t *timer);

struct tm* gmtime(const time_t *timer)
{
    return _gmtime32(timer);                                    
}
THE_END

  # Provide the mktime stub required by ffmpeg/avformat/utils.c.
  cat > mktime.c << 'THE_END'
/* Stub function for mktime.
 * This is an inline function in Visual C 2008 so is missing from msvcr90.dll
 */
#include <time.h>

time_t _mktime32(struct tm *timeptr);

time_t mktime(struct tm *timeptr)
{
    return _mktime32(timeptr);                                    
}
THE_END

  # Provide the localtime stub required by ffmpeg/avformat/utils.c.
  cat > localtime.c << 'THE_END'
/* Stub function for localtime.
 * This is an inline function in Visual C 2008 so is missing from msvcr90.dll
 */
#include <time.h>

struct tm *_localtime32(const time_t *timer);

struct tm *localtime(const time_t *timer)
{
    return _localtime32(timer);                                    
}
THE_END

  # Provide the _fstati64 stub required by ffmpeg/libavformat/file.c.
  cat > _fstati64.c << 'THE_END'
/* Stub function for _fstati64.
 * This is an alias in Visual C 2008 but has a fixed definition in MinGW
 */
#include <sys/stat.h>

int _fstat32i64(int fd, struct _stati64 *buffer);

int _fstati64(int fd, struct _stati64 *buffer)
{
    return _fstat32i64(fd, buffer);                                    
}
THE_END

  # Provide the time stub required by libavformat.
  cat > time.c << 'THE_END'
/* Stub function for time.
 * This is an inline function in Visual C 2008 so is missing from msvcr90.dll
 */
#include <time.h>

time_t _time32(time_t *timer);

time_t time(time_t *timer)
{
    return _time32(timer);                                    
}
THE_END


  gcc -c -O2 gmtime.c mktime.c localtime.c _fstati64.c time.c
  dlltool -d msvcr90.def -D msvcr90.dll -l libmsvcr90.dll.a
  ar rc libmsvcr90.dll.a gmtime.o mktime.o localtime.o _fstati64.o time.o
  ranlib libmsvcr90.dll.a
  cp -f libmsvcr90.dll.a "$BDMSVCR90"
  mv -f libmsvcr90.dll.a "$BDMSVCR90/libmsvcrt.dll.a"
  gcc -c -g gmtime.c mktime.c localtime.c _fstati64.c time.c
  dlltool -d msvcr90.def -D msvcr90d.dll -l libmsvcr90d.dll.a
  ar rc libmsvcr90d.dll.a gmtime.o mktime.o localtime.o _fstati64.o time.o
  ranlib libmsvcr90d.dll.a
  cp -f libmsvcr90d.dll.a "$BDMSVCR90"
  mv -f libmsvcr90d.dll.a "$BDMSVCR90/libmsvcrtd.dll.a"
  mv -f manifest.o "$BDMSVCR90"
  
  # These definitions are taken from mingw-runtime-3.12 .
  # The file was generated with the following command:
  #
  # gcc -DRUNTIME=msvcrt -D__FILENAME__=moldname-msvcrt.def
  #   -D__MSVCRT__ -C -E -P -xc-header moldname.def.in >moldname-msvcrt.def
  # It then had fstat deleted to match with msvcr90.dll.
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

  # Provide the fstat stub required by TIFF.
  cat > fstat.c << 'THE_END'
/* Stub function for fstat.
 * This is an inlined function in Visual C 2008 so is missing from msvcr90.dll
 */
#include <sys/stat.h>

int _fstat32(int fd, struct stat *buffer);

int fstat(int fd, struct stat *buffer)
{
    return _fstat32(fd, buffer);
}
THE_END

  # Provide the _winmajor constant required by libmingw32.a.
  cat > _winver.c << 'THE_END'
/* Windows Version constants.
 * Set to version 5.0 as msvcr90.dll cannot be used on any version before
 * Windows 2000.
*/
  /* Explicitly declare the pointers for linkage with dllimport */
  static unsigned int _winmajor = 5;
  static unsigned int _winminor = 0;
  static unsigned int _winver = 0x0500;
  unsigned int *_imp___winmajor = &_winmajor;
  unsigned int *_imp___winminor = &_winminor;
  unsigned int *_imp___winver = &_winver;
THE_END

  gcc -c -O2 fstat.c _winver.c
  ar x /mingw/lib/libmoldname90.a $OBJS
  dlltool --as as -k -U \
     --dllname msvcr90.dll \
     --def moldname-msvcrt.def \
     --output-lib libmoldname.dll.a
  ar rc libmoldname.dll.a $OBJS fstat.o _winver.o
  ranlib libmoldname.dll.a
  mv -f libmoldname.dll.a "$BDMSVCR90"
  gcc -c -g fstat.c _winver.c
  ar x /mingw/lib/libmoldname90d.a $OBJS
  dlltool --as as -k -U \
     --dllname msvcr90.dll \
     --def moldname-msvcrt.def \
     --output-lib libmoldnamed.dll.a
  ar rc libmoldnamed.dll.a $OBJS fstat.o _winver.o
  ranlib libmoldnamed.dll.a
  mv -f libmoldnamed.dll.a "$BDMSVCR90"
  rm -f ./*
  cd "$OLDPWD"
  rmdir /tmp/build_deps
fi
""")

alert = Preparation('computer beeper', r"""

#
# Alert the user by beeping the computer speaker
#
if [ x"$BDNTIMES" == x ]; then
  BDNTIMES=1
fi

for (( i = $BDNTIMES ; i ; i-- )); do
    printf $'\a\a\a\a\a'
    sleep 1s
done
""")

if __name__ == '__main__':
    sys.exit(main(dependencies, msvcr71_prep, msvcr90_prep, msys_prep))
