"""Config on Windows"""

import os, sys, shutil
from glob import glob

huntpaths = ['..', '..\\..', '..\\*', '..\\..\\*']


class Dependency:
    inc_hunt = ['include']
    lib_hunt = ['VisualC\\SDL\\Release', 'VisualC\\Release', 'Release', 'lib']
    def __init__(self, name, wildcard, lib, required = 0):
        self.name = name
        self.wildcard = wildcard
        self.required = required
        self.paths = []
        self.path = None
        self.inc_dir = None
        self.lib_dir = None
        self.lib = lib
        self.found = 0
        self.cflags = ''
                 
    def hunt(self):
        parent = os.path.abspath('..')
        for p in huntpaths:
            found = glob(os.path.join(p, self.wildcard))
            found.sort() or found.reverse()  #reverse sort
            for f in found:
                if f[:5] == '..'+os.sep+'..' and os.path.abspath(f)[:len(parent)] == parent:
                    continue
                if os.path.isdir(f):
                    self.paths.append(f)

    def choosepath(self):
        if not self.paths:
            print 'Path for ', self.name, 'not found.'
            if self.required: print 'Too bad that is a requirement! Hand-fix the "Setup"'
        elif len(self.paths) == 1:
            self.path = self.paths[0]
            print 'Path for '+self.name+':', self.path
        else:
            print 'Select path for '+self.name+':'
            for i in range(len(self.paths)):
                print '  ', i+1, '=', self.paths[i]
            print '  ', 0, '= <Nothing>'
            choice = raw_input('Select 0-'+`len(self.paths)`+' (1=default):')
            if not choice: choice = 1
            else: choice = int(choice)
            if(choice):
                self.path = self.paths[choice-1]

    def findhunt(self, base, paths):
        for h in paths:
            hh = os.path.join(base, h)
            if os.path.isdir(hh):
                return hh.replace('\\', '/')
        return base.replace('\\', '/')

    def configure(self):
        self.hunt()
        self.choosepath()
        if self.path:
            self.found = 1
            self.inc_dir = self.findhunt(self.path, Dependency.inc_hunt)
            self.lib_dir = self.findhunt(self.path, Dependency.lib_hunt)




DEPS = [
    Dependency('SDL', 'SDL-[0-9].*', 'SDL', 1),
    Dependency('FONT', 'SDL_ttf-[0-9].*', 'SDL_ttf'),
    Dependency('IMAGE', 'SDL_image-[0-9].*', 'SDL_image'),
    Dependency('MIXER', 'SDL_mixer-[0-9].*', 'SDL_mixer'),
#copy only dependencies
    Dependency('SMPEG', 'smpeg-[0-9].*', 'smpeg')
]


def setup_prebuilt():
    setup = open('Setup', 'w')
    for line in open('Setup.in').readlines():
        if line.startswith('#--'): continue
        if line.startswith('SDL = '):
            line = 'SDL = -Iprebuilt/include -Lprebuilt/lib -lSDL\n'
        setup.write(line)
    setup.write('COPYLIB_png $(SDL) -lzlib -llibpng1\n')


def main():
    if os.path.isdir('prebuilt'):
        reply = raw_input('\nUse the SDL libraries in "prebuilt"? [Y/n]')
        if not reply or reply[0].lower() != 'n':
            return setup_prebuilt()

    global DEPS
    for d in DEPS:
        d.configure()

    return DEPS    




if __name__ == '__main__':
    print """This is the configuration subscript for Windows.
Please run "config.py" for full configuration."""

