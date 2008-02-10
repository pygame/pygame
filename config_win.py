"""Config on Windows"""

import dll
import os, sys
from glob import glob
from distutils.sysconfig import get_python_inc

huntpaths = ['..', '..\\..', '..\\*', '..\\..\\*']


class Dependency(object):
    inc_hunt = ['include']
    lib_hunt = ['VisualC\\SDL\\Release', 'VisualC\\Release', 'Release', 'lib']
    def __init__(self, name, wildcards, libs=None, required = 0):
        if libs is None:
            libs = [dll.name_to_root(name)]
        self.name = name
        self.wildcards = wildcards
        self.required = required
        self.paths = []
        self.path = None
        self.inc_dir = None
        self.lib_dir = None
        self.libs = libs
        self.found = 0
        self.cflags = ''
                 
    def hunt(self):
        parent = os.path.abspath('..')
        for p in huntpaths:
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


class DependencyPython(object):
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
 
    def configure(self):
        self.found = 1
        if self.module:
            try:
                self.ver = __import__(self.module).__version__
            except ImportError:
                self.found = 0
        if self.found and self.header:
            fullpath = os.path.join(get_python_inc(0), self.header)
            if not os.path.isfile(fullpath):
                found = 0
            else:
                self.inc_dir = os.path.split(fullpath)[0]
        if self.found:
            print self.name + '        '[len(self.name):] + ': found', self.ver
        else:
            print self.name + '        '[len(self.name):] + ': not found'


class DependencyDLL(Dependency):
    def __init__(self, name=None, wildcards=None, link=None, libs=None):
        if libs is None:
            if name is not None:
                libs = [dll.name_to_root(name)]
            elif link is not None:
                libs = link.libs
            else:
                libs = []
        if name is None:
            name = link.name
        Dependency.__init__(self, 'COPYLIB_' + name, wildcards, libs)
        self.lib_name = name
        self.test = dll.tester(name)
        self.lib_dir = '_'
        self.found = 1
        self.link = link

    def configure(self):
        if self.link is None and self.wildcards:
            self.hunt()
            self.choosepath()
        else:
            self.path = self.link.path
        if self.path is not None:
            self.hunt_dll()

    def hunt_dll(self):
        for dir in self.lib_hunt:
            path = os.path.join(self.path, dir)
            try:
                entries = os.listdir(path)
            except:
                pass
            else:
                for e in entries:
                    if self.test(e) and os.path.isfile(os.path.join(path, e)):
                        # Found
                        self.lib_dir = os.path.join(path, e).replace('\\', '/')
                        print "DLL for %s is %s" % (self.lib_name, self.lib_dir)
                        return
        print "DLL for %s not found" % self.lib_name

                    
class DependencyWin(object):
    def __init__(self, name, libs):
        self.name = name
        self.inc_dir = None
        self.lib_dir = None
        self.libs = libs
        self.found = 1
        self.cflags = ''
        
    def configure(self):
        pass


DEPS = [
    Dependency('SDL', ['SDL-[1-9].*'], required=1),
    Dependency('FONT', ['SDL_ttf-[2-9].*']),
    Dependency('IMAGE', ['SDL_image-[1-9].*']),
    Dependency('MIXER', ['SDL_mixer-[1-9].*']),
    Dependency('SMPEG', ['smpeg-[0-9].*', 'smpeg']),
    DependencyWin('SCRAP', ['user32', 'gdi32']),
    Dependency('JPEG', ['jpeg-[6-9]*']),
    Dependency('PNG', ['libpng-[1-9].*']),
    DependencyDLL('TIFF', ['tiff-[3-9].*']),
    DependencyDLL('VORBIS', ['libvorbis-[1-9].*']),
    DependencyDLL('OGG', ['libogg-[1-9].*']),
    DependencyDLL('Z', ['zlib-[1-9].*']),
]

DEPS += [DependencyDLL(link=dep) for dep in DEPS[:] if type(dep) is Dependency]
DEPS += [DependencyDLL('VORBISFILE', link=DEPS[9])]

def setup_prebuilt():
    setup = open('Setup', 'w')
    for line in open('Setup.in').readlines():
        if line[:3] == '#--': continue
        if line[:6] == 'SDL = ':
            line = 'SDL = -Iprebuilt/include -Iprebuilt/include/SDL -Lprebuilt/lib -lSDL\n'
        if line[:8] == 'SMPEG = ':
            line = 'SMPEG = -Iprebuilt/include/smpeg -lsmpeg\n'
        if line[:8] == 'SCRAP = ':
            line = 'SCRAP = -luser32 -lgdi32\n'
        setup.write(line)


def main():
    if os.path.isdir('prebuilt'):
        reply = raw_input('\nUse the SDL libraries in "prebuilt"? [Y/n]')
        if not reply or reply[0].lower() != 'n':
            setup_prebuilt()
            raise SystemExit()

    global DEPS
    for d in DEPS:
        d.configure()
    
    return DEPS

if __name__ == '__main__':
    print """This is the configuration subscript for Windows.
Please run "config.py" for full configuration."""

