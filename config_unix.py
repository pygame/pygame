"""Config on Unix"""

import os, sys, shutil, string
from glob import glob

configcommand = os.environ.get('SDL_CONFIG', 'sdl-config')
configcommand = configcommand + ' --version --cflags --libs'
localbase = os.environ.get('LOCALBASE', '')


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
        
        for dir in incdirs:
            path = os.path.join(dir, incname)
            if os.path.isfile(path):
                self.inc_dir = dir
        for dir in libdirs:
            for name in libnames:
                path = os.path.join(dir, name)
                if os.path.isfile(path):
                    self.lib_dir = dir
                
        if self.lib_dir and self.inc_dir:
            print self.name + '        '[len(self.name):] + ': found'
            self.found = 1
        else:
            print self.name + '        '[len(self.name):] + ': not found'



sdl_lib_name = 'SDL'
if sys.platform.find('bsd') != -1:
    sdl_lib_name = 'SDL-1.2'

DEPS = [
    Dependency('SDL', 'SDL.h', 'lib'+sdl_lib_name+'.so', sdl_lib_name),
    Dependency('FONT', 'SDL_ttf.h', 'libSDL_ttf.so', 'SDL_ttf'),
    Dependency('IMAGE', 'SDL_image.h', 'libSDL_image.so', 'SDL_image'),
    Dependency('MIXER', 'SDL_mixer.h', 'libSDL_mixer.so', 'SDL_mixer'),
    Dependency('SMPEG', 'smpeg.h', 'libsmpeg.so', 'smpeg'),
]


def main():
    global DEPS
    
    print 'calling "sdl-config"'
    configinfo = "-I/usr/local/include/SDL -L/usr/local/lib -D_REENTRANT -lSDL"
    try:
        configinfo = os.popen(configcommand).readlines()
        print 'Found SDL version:', configinfo[0]
        configinfo = ' '.join(configinfo[1:])
        configinfo = configinfo.split()
        for w in configinfo[:]:
            if ',' in w: configinfo.remove(w)
        configinfo = ' '.join(configinfo)
        #print 'Flags:', configinfo
    except:
        raise SystemExit, """Cannot locate command, "sdl-config". Default SDL compile
flags have been used, which will likely require a little editing."""

    print 'Hunting dependencies...'
    if localbase:
        incdirs = [localbase + '/include/SDL']
        libdirs = [localbase + '/lib']
    else:
        incdirs = []
        libdirs = []
        for arg in configinfo.split():
            if arg[:2] == '-I':
                incdirs.append(arg[2:])
            elif arg[:2] == '-L':
                libdirs.append(arg[2:])
    for d in DEPS:
        d.configure(incdirs, libdirs)

    DEPS[0].inc_dirs = []
    DEPS[0].lib_dirs = []
    DEPS[0].cflags = configinfo

    return DEPS

    
if __name__ == '__main__':
    print """This is the configuration subscript for Unix.
Please run "config.py" for full configuration."""

