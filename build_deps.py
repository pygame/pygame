# Program build_deps.py

"""Build Pygame dependencies

Configured for Pygame 1.8 and Python 2.4 and up.

The libraries are installed in /usr/local of the MSYS directory structure.

This program can be run from a cmd or MSYS prompt. The current directory
and its outer directory are searched for the library source directories.

For the build to work the MSYS /usr/local header and library directories
must be part of the MinGW default search path. The necessary changes
to the mingw\\lib\\gcc\\mingw32\\<gcc verion #>\\specs file will be
made if the --prepare-mingw command line option is chosen. The specs file
is backed up as specs-original so the changes can be easily undone.

Python 2.4 and later are linked against msvcr71.dll. By default MinGW links
against the older msvcrt.dll. Unless the --no-msvcr71 option is closen the
--prepare-mingw option updates the MinGW specs file so MingGW links against
msvcr71.dll. Restoring the specs file undoes the changes.

Useful environment variables are "SHELL", the MSYS shell program (already defined
in the MSYS console), and "MINGW_ROOT_DIRECTORY". The program will prompt for any
missing information.

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
"""

from optparse import OptionParser
import os
import sys
import subprocess
from glob import glob
import time

#
#   Generic declarations
#
hunt_paths = ['.', '..']

# These functions are created by init()
#   windows_to_msys(path) => MSYS path
#   run_shell_script(script) => exit code

# This variable is set by init()
#   msys_root

def msys_raw_input(prompt=None):
    if prompt is not None:
        sys.stdout.flush()
        sys.stdout.write(prompt)
    sys.stdout.flush()
    return raw_input()

def confirm(message):
    "ask a yes/no question, return result"
    reply = msys_raw_input("\n%s [Y/n]:" % message)
    if reply and reply[0].lower() == 'n':
        return 0
    return 1

def as_flag(b):
    """Return bool b as a shell script flag '1' or '0'"""
    if b:
        return '1'
    return '0'

class Dependency(object):
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
            print "Path for %s: not found" % self.name
        elif len(self.paths) == 1:
            path = self.paths[0]
        else:
            print "Select path for %s:" % self.name
            for i in range(len(self.paths)):
                print "  %d = %s" % (i+1, self.paths[i])
            print "  0 = <Nothing>"
            choice = msys_raw_input("Select 0-%d (1=default):" % len(self.paths))
            if not choice:
                choice = 1
            else:
                choice = int(choice)
            if choice > 0:
                path = self.paths[choice-1]
        if path is not None:
            self.path = os.path.abspath(path)
            print "Path for %s: %s" % (self.name, self.path)

    def build(self):
        if self.path is not None:
            os.environ['BDWD'] = windows_to_msys(self.path)  # Is the path conversion overkill?
            return run_shell_script(self.shell_script)
        else:
            return None

class Preparation(object):
    def __init__(self, name, shell_script):
        self.name = name
        self.path = 'n/a'
        self.paths = []
        self.path = None
        self.dlls = []
        self.shell_script = shell_script

    def configure(self, hunt_paths):
        pass

    def build(self):
        return run_shell_script(self.shell_script)

def configure(dependencies):
    for dep in dependencies:
        dep.configure(hunt_paths)

def build(dependencies):
    results = []
    for dep in dependencies:
        results.append((dep.name, dep.build()))
        time.sleep(3)  # Give the subshells time to disappear
    return results

def input_shell():
    while 1:
        dir_path = msys_raw_input("Enter the MSYS directory path,\n"
                                  "(or press [Enter] to quit):")
        if not dir_path:
            return None
        roots = glob(os.path.join(dir_path, '[1-9].[0-9]'))
        roots.sort()
        roots.reverse()
        if not roots:
            print "\nNo msys versions found.\n"
        else:
            if len(roots) == 1:
                root = roots[0]
            else:
                print "Select an Msys version:"
                for i, path in enumerate(roots):
                    print "  %d = %s" % (i+1, os.path.split(path)[1])
                choice = msys_raw_input("Select 1-%d (1 = default):")
                if not choice:
                    root = roots[0]
                else:
                    root = roots[int(choice)-1]
            shell = os.path.join(os.path.abspath(root), 'bin', 'sh.exe')
            if os.path.isfile(shell):
                return shell
            else:
                print "\nThe specified Msys version has shell.\n"
    
def init(msys_directory=None, mingw_directory=None):
    """Set up environment"""
    # This function may terminate execution.
    global windows_to_msys, run_shell_script, msys_root

    def windows_to_msys(path):
        # Assumption: The path is absolute and starts with a drive letter.
        path_lower = path.lower()
        if path_lower.startswith(msys_root):
            return '/usr' + path[len(msys_root):].replace(os.sep, '/')
        if path_lower.startswith(mingw_root):
            return '/mingw' + path[len(mingw_root):].replace(os.sep, '/')
        drive, tail = os.path.splitdrive(path)
        return '/%s%s' % (drive[0], tail.replace(os.sep, '/'))

    def run_shell_script(script):
        cmd = [shell]
        if 'MSYSTEM' not in os.environ:
            cmd.append('--login')
        previous_cwd = os.getcwd()
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            p.communicate(script)
            return not bool(p.returncode)
        finally:
            os.chdir(previous_cwd)

    if mingw_directory is None:
        mingw_directory = '\\mingw'  # Default

    # Find the MSYS shell.
    try:
        file_path = os.environ['SHELL']
    except KeyError:
        if msys_directory is not None:
            file_path = os.path.join(msys_directory, 'bin', 'sh.exe')
        else:
            file_path = input_shell()
    shell = os.path.abspath(file_path)
    if not os.path.isfile(shell):
        shell = input_shell()
        if shell is None:
            sys.exit()
    msys_root = os.path.split(os.path.split(shell)[0])[0].lower()

    # Ensure MINGW_ROOT_DIRECTORY environment variable defined.
    try:
        dir_path = os.environ['MINGW_ROOT_DIRECTORY']
    except KeyError:
        dir_path = mingw_directory
    while 1:
        dir_path = os.path.abspath(dir_path)
        if os.path.isdir(dir_path):
            break
        dir_path = msys_raw_input("Enter the MINGW directory path,\n(or press [Enter] to quit):")
        if not dir_path:
            sys.exit()
    os.environ['MINGW_ROOT_DIRECTORY'] = dir_path
    mingw_root = dir_path.lower()

def main(dependencies, mingw_preparation, msys_preparation):
    names = [d.name for d in dependencies]
    usage = ("usage: %prog [options] --all\n"
             "       %prog [options] [args]\n"
             "\n"
             "Build the Pygame dependencies.  The args, if given, are\n"
             "libraries to include or exclude.\n"
             "\n"
             "At startup this program may prompt for missing information.\n"
             "Be aware of this before redirecting output or leaving the\n"
             "program unattended. Once the 'Starting build' message appears\n"
             "no more user input is required.\n"
             "\n"
             "Note that MSYS is unstable. A full build may hang.\n"
             "See --include and --help-args. Warning, the build\n"
             "will take awhile."
             "\n"
             "See the doc string at the beginning of the program for more details\n")
    
    parser = OptionParser(usage)
    parser.add_option('-a', '--all', action='store_true', dest='build_all',
                      help="Include all libraries in the build")
    parser.set_defaults(build_all=False)
    parser.add_option('-p', '--prepare-mingw', action='store_true', dest='prepare_mingw',
                      help="Make necessary changes to the MinGW specs file.\n"
                           "This only needs to be run once")
    parser.set_defaults(prepare_mingw=False)
    parser.add_option('--no-msvcr71', action='store_true', dest='msvcrt',
                      help="Do not link to msvcr71.dll; see --prepare-mingw")
    parser.set_defaults(msvcrt=False)
    parser.add_option('--no-configure', action='store_false', dest='configure',
                      help="Do not prepare the makefiles")
    parser.set_defaults(configure=True)
    parser.add_option('--no-compile', action='store_false', dest='compile',
                      help="Do not compile or install the libraries")
    parser.set_defaults(compile=True)
    parser.add_option('--no-install', action='store_false', dest='install',
                      help="Do not install the libraries")
    parser.set_defaults(install=True)
    parser.add_option('--clean', action='store_true', dest='clean',
                      help="Remove generated files (make clean) as a last step")
    parser.set_defaults(clean=False)
    parser.add_option('--clean-only', action='store_true', dest='clean_only',
                      help="Perform only a clean")
    parser.set_defaults(clean_only=False)
    parser.add_option('-e', '--exclude', action='store_true', dest='exclude',
                      help="Exclude the specified libraries")
    parser.set_defaults(exclude=False)
    parser.add_option('-m', '--msys-root', action='store', dest='msys_directory',
                      help="MSYS root directory path (which includes the 1.x subdirectory)")
    parser.set_defaults(msys_directory='')
    parser.add_option('-g', '--mingw-root', action='store', dest='mingw_directory',
                      help="MinGW root directory path")
    parser.set_defaults(mingw_directory='')
    parser.add_option('--help-args', action='store_true', dest='arg_help',
                      help="Show a list of recognised libraries, in build order, and exit")
    parser.set_defaults(arg_help=False)
    options, args = parser.parse_args()
    if options.arg_help:
        print "These are the Pygame library dependencies:"
        for n in names:
            print " ", n
        return 0
    if options.build_all:
        if args:
            print "No library names are accepted for the --all option."
            return 1
        if options.exclude:
            print "All libraries excluded"
            deps = []
        else:
            deps = dependencies
    elif args:
        args = [a.upper() for a in args]
        for a in args:
            if a not in names:
                print "%s is an unknown library; valid choices are:" % a
                for n in names:
                    print " ", n
                return 1
        if options.exclude:
            deps = [d for d in dependencies if d.name not in args]
        else:
            deps = [d for d in dependencies if d.name in args]
    else:
        print "No libraries specified."
        deps = []
    if options.prepare_mingw:
        deps.insert(0, mingw_preparation)
    if deps:
        deps.insert(0, msys_preparation)

    init(options.msys_directory, options.mingw_directory)
    configure(deps)
    print "=== Starting build ==="
    sys.stdout.flush()  # Make sure everything is displayed before scripts start.
    os.environ['BDCONF'] = as_flag(options.configure and not options.clean_only)
    os.environ['BDCOMP'] = as_flag(options.compile and not options.clean_only)
    os.environ['BDINST'] = as_flag(options.install and options.compile and not options.clean_only)
    os.environ['BDCLEAN'] = as_flag(options.clean or options.clean_only)
    os.environ['BDMSVCR71'] = as_flag(not options.msvcrt)
    start_time = time.time()
    results = build(deps)
    print "\n\n=== Summary ==="
    for name, result in results:
        if result is not None and not result:
            print "  %s reported errors" % name
    bin_dir = os.path.join(msys_root, 'local', 'bin')
    print
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
            print "  %-10s: %s" % (name, msg)
    
#
#   Build specific code
#

# This list includes the MSYS shell scripts to build each library.
# Each script runs in an environment where MINGW_ROOT_DIRECTORY is
# defined and the MinGW bin directory is in PATH. Three other environment
# variables are defined: BDCONF, BDCOMP, BDINST and BDCLEAN. They are either
# '0' or '1'. They represent configure, compile, install and clean respectively.
# When '1' the corresponding action is performed. When '0' it is skipped.
# A final variable, DBWD, is the root directory of the source code. A
# script will cd to it before doing anything else.
# 
# The list order corresponds to build order. It is critical.
dependencies = [
    Dependency('SDL', ['SDL-[1-9].*'], ['SDL.dll'], """

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
  BDDXHDR=

  # Build and install as win32 gui.
  export LDFLAGS='-mwindows'
  ./configure
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCOMP == x1 ]; then
  make
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDINST == x1 ]; then
  make install
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCLEAN == x1 ]; then
  make clean
fi
"""),
    Dependency('Z', ['zlib-[1-9].*'], ['zlib1.dll'], """

cd $BDWD

if [ x$BDCONF == x1 ]; then
  # Use the existing gcc makefile, modified to build a win32 gui.
  sed 's/dllwrap/dllwrap -mwindows/' win32/Makefile.gcc >Makefile.gcc
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCOMP == x1 ]; then
  # Build with the import library renamed.
  make IMPLIB='libz.dll.a' -fMakefile.gcc
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDINST == x1 ]; then
  # Have to do own install.
  cp -fp *.a /usr/local/lib
  cp -fp zlib.h /usr/local/include
  cp -fp zconf.h /usr/local/include
  cp -fp zlib1.dll /usr/local/bin
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCLEAN == x1 ]; then
  make clean
fi
"""),
    Dependency('FREETYPE', ['freetype-[2-9].*'], [], """

cd $BDWD

if [ x$BDCONF == x1 ]; then
  # Will only install a static library, but that is all that is needed.
  ./configure --disable-shared
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCOMP == x1 ]; then
  make
  if [ ! -f objs/.libs/libfreetype.a ]; then exit 1; fi
fi

if [ x$BDINST == x1 ]; then
  make install
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCLEAN == x1 ]; then
  make clean
fi
"""),
    Dependency('FONT', ['SDL_ttf-[2-9].*'], ['SDL_ttf.dll'], """

cd $BDWD

if [ x$BDCONF == x1 ]; then
  ./configure
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCOMP == x1 ]; then
  make
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDINST == x1 ]; then
  make install
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCLEAN == x1 ]; then
  make clean
fi
"""),
    Dependency('PNG', ['libpng-[1-9].*'], ['libpng13.dll'], """

cd $BDWD

if [ x$BDCONF == x1 ]; then
  # This will only build a static library.
  ./configure --with-libpng-compat=no --disable-shared
  if [ x$? != x0 ]; then exit $?; fi

  # Remove a duplicate entry in the def file.
  sed '222 d' scripts/pngw32.def >in.def
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCOMP == x1 ]; then
  # Build the DLL as win32 gui.
  make
  if [ x$? != x0 ]; then exit $?; fi
  dlltool -D libpng13.dll -d in.def -l libpng.dll.a
  if [ x$? != x0 ]; then exit $?; fi
  ranlib libpng.dll.a
  if [ x$? != x0 ]; then exit $?; fi
  gcc -shared -s -mwindows -def in.def -o libpng13.dll .libs/libpng12.a -lz
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDINST == x1 ]; then
  # Only install the headers and import library, otherwise SDL_image will
  # statically link to png.
  make install-pkgincludeHEADERS
  cp -fp png.h /usr/local/include
  cp -fp pngconf.h /usr/local/include
  cp -fp libpng.dll.a /usr/local/lib
  cp -fp libpng13.dll /usr/local/bin
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCLEAN == x1 ]; then
  make clean
  rm -f in.def
  rm -f libpng.dll.a
  rm -f libpng13.dll
fi
"""),
    Dependency('JPEG', ['jpeg-[6-9]*'], ['jpeg.dll'], """

cd $BDWD

if [ x$BDCONF == x1 ]; then
  # This will only build a static library.
  ./configure --disable-shared
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCOMP == x1 ]; then
  # Build the DLL as a win32 gui.
  make CFLAGS='-O2'
  if [ x$? != x0 ]; then exit $?; fi
  dlltool --export-all-symbols -D jpeg.dll -l libjpeg.dll.a -z in.def libjpeg.a
  if [ x$? != x0 ]; then exit $?; fi
  ranlib libjpeg.dll.a
  if [ x$? != x0 ]; then exit $?; fi
  gcc -shared -s -mwindows -def in.def -o jpeg.dll libjpeg.a
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDINST == x1 ]; then
  # Only install the headers and import library, otherwise SDL_image will
  # statically link to jpeg.
  make install-headers
  cp -fp libjpeg.dll.a /usr/local/lib
  cp -fp jpeg.dll /usr/local/bin
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCLEAN == x1 ]; then
  make clean
  rm -f in.def
  rm -f libjpeg.dll.a
  rm -f jpeg.dll
fi
"""),
    Dependency('TIFF', ['tiff-[3-9].*'], ['libtiff.dll'], """

cd $BDWD

if [ x$BDCONF == x1 ]; then
  # The shared library build does not work
  ./configure --disable-cxx --prefix=/usr/local --disable-shared
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCOMP == x1 ]; then
  make
  if [ x$? != x0 ]; then exit $?; fi

  # Build the DLL as a win32 gui
  cd libtiff
  gcc -shared -s -mwindows -def libtiff.def -o libtiff.dll .libs/libtiff.a -ljpeg -lz
  if [ x$? != x0 ]; then exit $?; fi
  dlltool -D libtiff.dll -d libtiff.def -l libtiff.dll.a
  if [ x$? != x0 ]; then exit $?; fi
  ranlib libtiff.dll.a
  if [ x$? != x0 ]; then exit $?; fi
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

if [ x$BDCLEAN == x1 ]; then
  make clean
  rm -f libtiff/libtiff.dll.a
  rm -f libtiff/libtiff.dll
fi
"""),
    Dependency('IMAGE', ['SDL_image-[1-9].*'], ['SDL_image.dll'], """

cd $BDWD

if [ x$BDCONF == x1 ]; then
  # Disable dynamic loading of image libraries as that uses wrong DLL search path
  ./configure --disable-jpeg-shared --disable-png-shared --disable-tif-shared
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCOMP == x1 ]; then
  make
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDINST == x1 ]; then
  make install
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCLEAN == x1 ]; then
  make clean
fi
"""),
    Dependency('SMPEG', ['smpeg-[0-9].*', 'smpeg'], ['smpeg.dll'], """

cd $BDWD

if [ x$BDCONF == x1 ]; then
  # This comes straight from SVN so has no configure script
  ./autogen.sh
  if [ x$? != x0 ]; then exit $?; fi
  ./configure --disable-gtk-player --disable-opengl-player --disable-gtktest
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCOMP == x1 ]; then
  make CXXLD='$(CXX) -no-undefined'
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDINST == x1 ]; then
  make install
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCLEAN == x1 ]; then
  make clean
fi
"""),
    Dependency('OGG', ['libogg-[1-9].*'], ['libogg-0.dll'], """

cd $BDWD

if [ x$BDCONF == x1 ]; then
  # Build as win32 gui.
  export LDFLAGS='-mwindows'
  ./configure
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCOMP == x1 ]; then
  make
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDINST == x1 ]; then
  make install
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCLEAN == x1 ]; then
  make clean
fi
"""),
    Dependency('VORBIS', ['libvorbis-[1-9].*'], ['libvorbis-0.dll', 'libvorbisfile-3.dll'], """

cd $BDWD

if [ x$BDCONF == x1 ]; then
  # Build as win32 gui.
  export LDFLAGS='-mwindows'
  ./configure
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCOMP == x1 ]; then
  make LIBS='-logg'
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDINST == x1 ]; then
  make install
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCLEAN == x1 ]; then
  make clean
fi
"""),
    Dependency('MIXER', ['SDL_mixer-[1-9].*'], ['SDL_mixer.dll'], """

cd $BDWD

if [ x$BDCONF == x1 ]; then
  # Remove INCLUDE or configure script fails
  export INCLUDE=''
  ./configure --disable-music-ogg-shared --disable-music-mp3-shared
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCOMP == x1 ]; then
  make
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDINST == x1 ]; then
  make install
  if [ x$? != x0 ]; then exit $?; fi
fi

if [ x$BDCLEAN == x1 ]; then
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
    
mingw_prep = Preparation('MinGW Preparation', r"""

set -e

#
#   msvcr71.dll support
#
if [ x$BDMSVCR71 == x1 ]; then
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

  ar x /mingw/lib/libmoldname.a $OBJS
  dlltool --as as -k -U \
     --dllname msvcr71.dll \
     --def moldname-msvcrt.def \
     --output-lib libmoldname71.a
  ar rc libmoldname71.a $OBJS
  ranlib libmoldname71.a
  cp -fp libmoldname71.a /mingw/lib
  # In the default libraries list replace msvcrt libs with msvcrt71 versions
  SEDOPTSVC="
    s/-lmsvcrt/-lmsvcr71/;
    s/-lmoldname /-lmoldname71 /;"
  rm -f ./*
  cd $OLDPWD
  rmdir /tmp/build_deps
else
  SEDOPTSVC=
fi

#
#   Add /usr/local to MinGW searches.
#
# Get /usr/local as a Windows path with Unix path separators
# and properly escaped for sed
LOCALDIR=`python -c 'import sys; print sys.argv[1].replace("/", "\/")' /usr/local`
# Append include and library paths to the appropriate gcc options lists
SEDOPTSLD="
  /^\*cpp:\$/{N; s/\(.\)$/\1 -I$LOCALDIR\/include/;};
  /^\*link:\$/{N; s/\(.\)$/\1 -L$LOCALDIR\/lib/;};"

#
#   Modify the specs file
#
GCCVER=`gcc -dumpversion`
SPECDIR="/mingw/lib/gcc/mingw32/$GCCVER"

# Make a backup if one does not exist, else restore specs.
if [ ! -f $SPECDIR/specs-original ]; then
  cp $SPECDIR/specs $SPECDIR/specs-original
else
  cp $SPECDIR/specs-original $SPECDIR/specs
fi

SEDOPTS="$SEDOPTSLD $SEDOPTSVC"
SEDADDCR=$'s/\\(.*\\)$/\\1\r/;'
tr -d \\r <$SPECDIR/specs-original | sed "$SEDOPTS" | sed "$SEDADDCR" >$SPECDIR/specs
""")

if __name__ == '__main__':
    sys.exit(main(dependencies, mingw_prep, msys_prep))
