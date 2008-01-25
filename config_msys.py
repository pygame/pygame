"""Config on Msys mingw"""

import dll
import os, sys, string
import subprocess
import re
from glob import glob
from distutils.sysconfig import get_python_inc
import itertools

configcommand = os.environ.get('SDL_CONFIG', 'sdl-config',)
configcommand = configcommand + ' --version --cflags --libs'
localbase = os.environ.get('LOCALBASE', '')

#these get prefixes with '/usr/local' and /mingw or the $LOCALBASE
origincdirs = ['/include', '/include/SDL', '/include/SDL11',
               '/include/smpeg' ]
origlibdirs = ['/lib']


class ConfigError(Exception):
    pass

msys_root = os.path.split(os.path.split(os.environ['SHELL'])[0])[0]  # Get msys root directory
try:
    mingw_root = os.environ['MINGW_ROOT_DIRECTORY']
except KeyError:
    pass
check_for_drive = re.compile('/[A-Z]/', re.I).match
def has_drive(path):
    return check_for_drive(path) is not None

class ConvertionError(ConfigError):
    pass

def msys_to_windows(path):
    """Return a Windows translation of an MSYS path

    The Unix path separator is uses as it survives the distutils setup file read process.
    """
    if path.startswith('/usr'):
        path =  msys_root + path[4:]
    elif path.startswith('/mingw'):
        try:
            path =  mingw_root + path[6:]
        except NameError:
            raise ConversionError('MINGW_ROOT_DIRECTORY environment variable undefined')
    elif has_drive(path):
        path =  path[1] + ":" + path[2:]
    elif path == '/':
        path = msys_root
    elif path.startswith('/'):
        path =  msys_root + path
    return path.replace(os.sep, '/')

def path_join(a, *p):
    return os.path.join(a, *p).replace(os.sep, '/')
path_split = os.path.split

def confirm(message):
    "ask a yes/no question, return result"
    #The output must be flushed for the prompt to be visible on MSYS bash
    sys.stdout.write("\n%s [Y/n]:" % message)
    sys.stdout.flush()
    reply = raw_input()
    if reply and string.lower(reply[0]) == 'n':
        return 0
    return 1

class DependencyProg:
    needs_dll = True
    def __init__(self, name, envname, exename, minver, defaultlibs):
        self.name = name
        command = os.environ.get(envname, exename)
        drv, pth = os.path.splitdrive(command)
        if drv:
            command = '/' + drv[0] + pth
        shell = os.environ['SHELL']
        self.lib_dir = ''
        self.inc_dir = ''
        self.libs = []
        self.cflags = ''
        try:
            config = subprocess.Popen([shell, command, '--version', '--cflags', '--libs'],
                                      stdout=subprocess.PIPE
                                      ).communicate()[0]
            ver, flags = config.split('\n', 1)
            self.ver = ver.strip()
            flags = flags.split()
            if minver and self.ver < minver:
                err= 'WARNING: requires %s version %s (%s found)' % (self.name, self.ver, minver)
                raise ValueError, err
            self.found = 1
            self.cflags = ''
            for f in flags:
                if f[:2] in ('-I', '-L'):
                    self.cflags += f[:2] + msys_to_windows(f[2:]) + ' '
                elif f[:2] in ('-l', '-D'):
                    self.cflags += f + ' '
                elif f[:3] == '-Wl':
                    self.cflags += '-Xlinker ' + f + ' '
        except:
            print 'WARNING: "%s" failed!' % command    
            self.found = 0
            self.ver = '0'
            self.libs = defaultlibs

    def configure(self, incdirs, libdir):
        if self.found:
            print self.name + '        '[len(self.name):] + ': found ' + self.ver
            self.found = 1
        else:
            print self.name + '        '[len(self.name):] + ': not found'

class Dependency:
    needs_dll = True
    def __init__(self, name, checkhead, checklib, libs):
        self.name = name
        self.inc_dir = None
        self.lib_dir = None
        self.libs = libs
        self.found = 0
        self.checklib = checklib
        self.checkhead = checkhead
        self.cflags = ''
    
    def configure(self, incdirs, libdirs):
        self.find_inc_dir(incdirs)
        self.find_lib_dir(libdirs)
        
        if self.lib_dir and self.inc_dir:
            print self.name + '        '[len(self.name):] + ': found'
            self.found = 1
        else:
            print self.name + '        '[len(self.name):] + ': not found'

    def find_inc_dir(self, incdirs):
        incname = self.checkhead
        for dir in incdirs:
            path = path_join(dir, incname)
            if os.path.isfile(path):
                self.inc_dir = dir
                return

    def find_lib_dir(self, libdirs):
        libname = self.checklib
        for dir in libdirs:
            path = path_join(dir, libname)
            if filter(os.path.isfile, glob(path+'*')):
                self.lib_dir = dir
                return

        
class DependencyPython:
    needs_dll = False
    def __init__(self, name, module, header):
        self.name = name
        self.lib_dir = ''
        self.inc_dir = ''
        self.libs = []
        self.cflags = ''
        self.found = 0
        self.ver = '0'
        self.module = module
        self.header = header
 
    def configure(self, incdirs, libdirs):
        self.found = 1
        if self.module:
            try:
                self.ver = __import__(self.module).__version__
            except ImportError:
                self.found = 0
        if self.found and self.header:
            fullpath = path_join(get_python_inc(0), self.header)
            if not os.path.isfile(fullpath):
                self.found = 0
            else:
                self.inc_dir = os.path.split(fullpath)[0]
        if self.found:
            print self.name + '        '[len(self.name):] + ': found', self.ver
        else:
            print self.name + '        '[len(self.name):] + ': not found'

class DependencyWin:
    needs_dll = False
    def __init__(self, name, libs):
        self.name = name
        self.inc_dir = None
        self.lib_dir = None
        self.libs = libs
        self.found = 1
        self.cflags = ''
        
    def configure(self, incdirs, libdirs):
        pass

class DependencyDLL:
    needs_dll = False
    def __init__(self, name):
        self.name = 'DLL_' + name
        self.inc_dir = None
        self.lib_dir = '_'
        self.libs = []
        self.found = 1  # Alway found to make its COPYLIB work
        self.cflags = ''
        self.lib_name = name
        self.file_name_test = dll.tester(name)

    def configure(self, incdirs, libdirs, start=None):
        omit = []
        if start is not None:
            if self.set_path(start):
                return
            omit.append(start)
            p, f = path_split(start)
            if f == 'lib' and self.set_path(path_join(p, 'bin')):
                return
            omit.append(start)
        # Search other directories
        for dir in libdirs:
            if dir not in omit:
                if self.set_path(dir):
                    return
                p, f = path_split(dir)
                if f == 'lib' and self.set_path(path_join(p, 'bin')):  # cond. and
                    return

    def set_path(self, wdir):
        test = self.file_name_test
        try:
            files = os.listdir(wdir)
        except:
            pass
        else:
            for f in files:
                if test(f) and os.path.isfile(path_join(wdir, f)):
                    # Found
                    self.lib_dir = path_join(wdir, f)
                    return True
        # Not found
        return False


def main():
    print '\nHunting dependencies...'
    DEPS = [
        DependencyProg('SDL', 'SDL_CONFIG', 'sdl-config', '1.2', ['SDL']),
        Dependency('FONT', 'SDL_ttf.h', 'libSDL_ttf.dll.a', ['SDL_ttf']),
        Dependency('IMAGE', 'SDL_image.h', 'libSDL_image.dll.a', ['SDL_image']),
        Dependency('MIXER', 'SDL_mixer.h', 'libSDL_mixer.dll.a', ['SDL_mixer']),
        DependencyProg('SMPEG', 'SMPEG_CONFIG', 'smpeg-config', '0.4.3', ['smpeg']),
        Dependency('PNG', 'png.h', 'libpng.dll.a', ['png']),
        Dependency('JPEG', 'jpeglib.h', 'libjpeg.dll.a', ['jpeg']),
        DependencyWin('SCRAP', ['user32', 'gdi32']),
        DependencyDLL('TIFF'),
        DependencyDLL('VORBISFILE'),
        DependencyDLL('VORBIS'),
        DependencyDLL('OGG'),
        DependencyDLL('Z'),
    ]

    if not DEPS[0].found:
        print 'Unable to run "sdl-config". Please make sure a development version of SDL is installed.'
        raise SystemExit

    if localbase:
        incdirs = [localbase+d for d in origincdirs]
        libdirs = [localbase+d for d in origlibdirs]
    else:
        incdirs = []
        libdirs = []
    incdirs += [msys_to_windows("/usr/local"+d) for d in origincdirs]
    libdirs += [msys_to_windows("/usr/local"+d) for d in origlibdirs]
    incdirs += [msys_to_windows("/mingw"+d) for d in origincdirs]
    libdirs += [msys_to_windows("/mingw"+d) for d in origlibdirs]
    for arg in string.split(DEPS[0].cflags):
        if arg[:2] == '-I':
            incdirs.append(arg[2:])
        elif arg[:2] == '-L':
            libdirs.append(arg[2:])
    dll_deps = []
    for d in DEPS:
        d.configure(incdirs, libdirs)
        if d.needs_dll:
            dll_dep = DependencyDLL(d.name)
            dll_dep.configure(incdirs, libdirs, d.lib_dir)
            dll_deps.append(dll_dep)

    DEPS += dll_deps
    for d in DEPS:
        if isinstance(d, DependencyDLL):
            if d.lib_dir == '':
                print "DLL for %-12s: not found" % d.lib_name
            else:
                print "DLL for %-12s: %s" % (d.lib_name, d.lib_dir)
    
    for d in DEPS[1:]:
        if not d.found:
            if not confirm("""
Warning, some of the pygame dependencies were not found. Pygame can still
compile and install, but games that depend on those missing dependencies
will not run. Would you like to continue the configuration?"""):
                raise SystemExit()
            break

    return DEPS

if __name__ == '__main__':
    print """This is the configuration subscript for MSYS.
Please run "config.py" for full configuration."""

