"""Config on Unix"""

import os, sys
from glob import glob
from distutils.sysconfig import get_python_inc

# Python 2.x/3.x compatibility
try:
    raw_input
except NameError:
    raw_input = input

configcommand = os.environ.get('SDL_CONFIG', 'sdl-config',)
configcommand = configcommand + ' --version --cflags --libs'
localbase = os.environ.get('LOCALBASE', '')

#these get prefixes with '/usr' and '/usr/local' or the $LOCALBASE
origincdirs = ['/include', '/include/SDL', '/include/SDL']
origlibdirs = ['/lib','/lib64','/X11R6/lib']
if 'ORIGLIBDIRS' in os.environ and os.environ['ORIGLIBDIRS'] != "":
    origlibdirs = os.environ['ORIGLIBDIRS'].split(":")


def confirm(message):
    "ask a yes/no question, return result"
    if not sys.stdout.isatty():
        return False
    reply = raw_input('\n' + message + ' [Y/n]:')
    if reply and (reply[0].lower()) == 'n':
        return False
    return True

class DependencyProg:
    def __init__(self, name, envname, exename, minver, defaultlibs, version_flag="--version"):
        self.name = name
        command = os.environ.get(envname, exename)
        self.lib_dir = ''
        self.inc_dir = ''
        self.libs = []
        self.cflags = ''
        try:
            # freetype-config for freetype2 version 2.3.7 on Debian lenny
            # does not recognize multiple command line options. So execute
            # 'command' separately for each option.
            config = (os.popen(command + ' ' + version_flag).readlines() +
                      os.popen(command + ' --cflags').readlines() +
                      os.popen(command + ' --libs').readlines())
            flags = ' '.join(config[1:]).split()

            # remove this GNU_SOURCE if there... since python has it already,
            #   it causes a warning.
            if '-D_GNU_SOURCE=1' in flags:
                flags.remove('-D_GNU_SOURCE=1')
            self.ver = config[0].strip()
            if minver and self.ver < minver:
                err= 'WARNING: requires %s version %s (%s found)' % (self.name, self.ver, minver)
                raise ValueError(err)
            self.found = 1
            self.cflags = ''
            for f in flags:
                if f[:2] in ('-l', '-D', '-I', '-L'):
                    self.cflags += f + ' '
                elif f[:3] == '-Wl':
                    self.cflags += '-Xlinker ' + f + ' '
            if self.name == 'SDL':
                inc = '-I' + '/usr/X11R6/include'
                self.cflags = inc + ' ' + self.cflags
        except:
            print ('WARNING: "%s" failed!' % command)
            self.found = 0
            self.ver = '0'
            self.libs = defaultlibs

    def configure(self, incdirs, libdir):
        if self.found:
            print (self.name + '        '[len(self.name):] + ': found ' + self.ver)
            self.found = 1
        else:
            print (self.name + '        '[len(self.name):] + ': not found')

class Dependency:
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
        incname = self.checkhead
        libnames = self.checklib, self.name.lower()
        
        if incname:
            for dir in incdirs:
                path = os.path.join(dir, incname)
                if os.path.isfile(path):
                    self.inc_dir = dir

        for dir in libdirs:
            for name in libnames:
                path = os.path.join(dir, name)
                if filter(os.path.isfile, glob(path+'*')):
                    self.lib_dir = dir

        if (incname and self.lib_dir and self.inc_dir) or (not incname and self.lib_dir):
            print (self.name + '        '[len(self.name):] + ': found')
            self.found = 1
        else:
            print (self.name + '        '[len(self.name):] + ': not found')


class DependencyPython:
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
            fullpath = os.path.join(get_python_inc(0), self.header)
            if not os.path.isfile(fullpath):
                self.found = 0
            else:
                self.inc_dir = os.path.split(fullpath)[0]
        if self.found:
            print (self.name + '        '[len(self.name):] + ': found', self.ver)
        else:
            print (self.name + '        '[len(self.name):] + ': not found')

sdl_lib_name = 'SDL'

def main():
    print ('\nHunting dependencies...')
    DEPS = [
        DependencyProg('SDL', 'SDL_CONFIG', 'sdl-config', '1.2', ['sdl']),
        Dependency('FONT', 'SDL_ttf.h', 'libSDL_ttf.so', ['SDL_ttf']),
        Dependency('IMAGE', 'SDL_image.h', 'libSDL_image.so', ['SDL_image']),
        Dependency('MIXER', 'SDL_mixer.h', 'libSDL_mixer.so', ['SDL_mixer']),
        Dependency('PNG', 'png.h', 'libpng', ['png']),
        Dependency('JPEG', 'jpeglib.h', 'libjpeg', ['jpeg']),
        Dependency('SCRAP', '', 'libX11', ['X11']),
        Dependency('PORTMIDI', 'portmidi.h', 'libportmidi.so', ['portmidi']),
        Dependency('PORTTIME', 'porttime.h', 'libporttime.so', ['porttime']),
        DependencyProg('FREETYPE', 'FREETYPE_CONFIG', 'freetype-config', '2.0', ['freetype'], '--ftversion'),
        #Dependency('GFX', 'SDL_gfxPrimitives.h', 'libSDL_gfx.so', ['SDL_gfx']),
    ]
    if not DEPS[0].found:
        sys.exit('Unable to run "sdl-config". Please make sure a development version of SDL is installed.')

    incdirs = []
    libdirs = []
    incdirs += ["/usr"+d for d in origincdirs]
    libdirs += ["/usr"+d for d in origlibdirs]
    incdirs += ["/usr/local"+d for d in origincdirs]
    libdirs += ["/usr/local"+d for d in origlibdirs]
    if localbase:
        incdirs = [localbase+d for d in origincdirs]
        libdirs = [localbase+d for d in origlibdirs]

    for arg in DEPS[0].cflags.split():
        if arg[:2] == '-I':
            incdirs.append(arg[2:])
        elif arg[:2] == '-L':
            libdirs.append(arg[2:])
    for d in DEPS:
        d.configure(incdirs, libdirs)

    for d in DEPS[1:]:
        if not d.found:
            if "-auto" not in sys.argv and not confirm("""
Warning, some of the pygame dependencies were not found. Pygame can still
compile and install, but games that depend on those missing dependencies
will not run. Would you like to continue the configuration?"""):
                raise SystemExit("Missing dependencies")
            break

    return DEPS

if __name__ == '__main__':
    print ("""This is the configuration subscript for Unix.
Please run "config.py" for full configuration.""")

