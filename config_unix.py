"""Config on Unix"""

import os, sys, string
from glob import glob
from distutils.sysconfig import get_python_inc

configcommand = os.environ.get('SDL_CONFIG', 'sdl-config',)
configcommand = configcommand + ' --version --cflags --libs'
localbase = os.environ.get('LOCALBASE', '')

#these get prefixes with '/usr' and '/usr/local' or the $LOCALBASE
origincdirs = ['/include', '/include/SDL', '/include/SDL11',
               '/include/smpeg' ]
origlibdirs = ['/lib','/lib64']



def confirm(message):
    "ask a yes/no question, return result"
    reply = raw_input('\n' + message + ' [Y/n]:')
    if reply and string.lower(reply[0]) == 'n':
        return 0
    return 1




class DependencyProg:
    def __init__(self, name, envname, exename, minver, defaultlib):
        self.name = name
        command = os.environ.get(envname, exename)
        self.lib_dir = ''
        self.inc_dir = ''
        self.lib = ''
        self.cflags = ''
        try:
            config = os.popen(command + ' --version --cflags --libs').readlines()
            flags = string.split(string.join(config[1:], ' '))
            self.ver = string.strip(config[0])
            if minver and self.ver < minver:
                err= 'WARNING: requires %s version %s (%s found)' % (self.name, self.ver, minver)
                raise ValueError, err
            self.found = 1
            self.cflags = ''
            for f in flags:
                #if f[:2] == '-L':
                #    self.lib_dir += f[2:] + ' '
                #elif f[:2] == 'I':
                #    self.inc_dir += f[2:] + ' '
                if f[:2] in ('-l', '-D', '-I', '-L'):
                    self.cflags += f + ' '
                elif f[:3] == '-Wl':
                    self.cflags += '-Xlinker ' + f + ' '
        except:
            print 'WARNING: "%s" failed!' % command    
            self.found = 0
            self.ver = '0'
            self.lib = defaultlib

    def configure(self, incdirs, libdir):
        if self.found:
            print self.name + '        '[len(self.name):] + ': found ' + self.ver
            self.found = 1
        else:
            print self.name + '        '[len(self.name):] + ': not found'

                    
class Dependency:
    def __init__(self, name, checkhead, checklib, lib):
        self.name = name
        self.inc_dir = None
        self.lib_dir = None
        self.lib = lib
        self.found = 0
        self.checklib = checklib
        self.checkhead = checkhead
        self.cflags = ''
    
    def configure(self, incdirs, libdirs):
        incname = self.checkhead
        libnames = self.checklib, string.lower(self.name)
        
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
            print self.name + '        '[len(self.name):] + ': found'
            self.found = 1
        else:
            print self.name + '        '[len(self.name):] + ': not found'

class DependencyPython:
    def __init__(self, name, module, header):
        self.name = name
        self.lib_dir = ''
        self.inc_dir = ''
        self.lib = ''
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
            print self.name + '        '[len(self.name):] + ': found', self.ver
        else:
            print self.name + '        '[len(self.name):] + ': not found'



sdl_lib_name = 'SDL'
if sys.platform.find('bsd') != -1:
    sdl_lib_name = 'SDL-1.1'


def main():
    print '\nHunting dependencies...'
    DEPS = [
        DependencyProg('SDL', 'SDL_CONFIG', 'sdl-config', '1.2', 'sdl'),
        Dependency('FONT', 'SDL_ttf.h', 'libSDL_ttf.so', 'SDL_ttf'),
        Dependency('IMAGE', 'SDL_image.h', 'libSDL_image.so', 'SDL_image'),
        Dependency('MIXER', 'SDL_mixer.h', 'libSDL_mixer.so', 'SDL_mixer'),
        DependencyProg('SMPEG', 'SMPEG_CONFIG', 'smpeg-config', '0.4.3', 'smpeg'),
        DependencyPython('NUMERIC', 'Numeric', 'Numeric/arrayobject.h'),
        Dependency('PNG', 'png.h', 'libpng', 'png'),
        Dependency('JPEG', 'jpeglib.h', 'libjpeg', 'jpeg'),
        Dependency('X11', '', 'libX11', 'X11'),
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
    incdirs += ["/usr"+d for d in origincdirs]
    libdirs += ["/usr"+d for d in origlibdirs]
    incdirs += ["/usr/local"+d for d in origincdirs]
    libdirs += ["/usr/local"+d for d in origlibdirs]

    # some stuff for X11 on freebsd.
    incdirs += ["/usr/X11R6"+d for d in origincdirs]
    libdirs += ["/usr/X11R6"+d for d in origlibdirs]

    for arg in string.split(DEPS[0].cflags):
        if arg[:2] == '-I':
            incdirs.append(arg[2:])
        elif arg[:2] == '-L':
            libdirs.append(arg[2:])
    for d in DEPS:
        d.configure(incdirs, libdirs)


    for d in DEPS[1:]:
        if not d.found:
            if not confirm("""
Warning, some of the pygame dependencies were not found. Pygame can still
compile and install, but games that depend on those missing dependencies
will not run. Would you like to continue the configuration?"""):
                raise SystemExit
            break

    return DEPS

    
if __name__ == '__main__':
    print """This is the configuration subscript for Unix.
Please run "config.py" for full configuration."""

