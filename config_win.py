"""Config on Windows"""

# **** The search part is broken. For instance, the png Visual Studio project
# places to dll in a directory not checked by this module.

from setup_win_common import get_definitions

import os, sys
import re
from glob import glob
from distutils.sysconfig import get_python_inc

try:
    raw_input
except NameError:
    raw_input = input

huntpaths = ['..', '..\\..', '..\\*', '..\\..\\*']

def get_ptr_size():
    return 64 if sys.maxsize > 2**32 else 32

def as_machine_type(size):
    """Return pointer bit size as a Windows machine type"""
    if size == 32:
        return "x86"
    if size == 64:
        return "x64"
    raise BuildError("Unknown pointer size {}".format(size))

class Dependency(object):
    inc_hunt = ['include']
    lib_hunt = ['VisualC\\SDL\\Release', 'VisualC\\Release', 'Release', 'lib']
    def __init__(self, name, wildcards, libs=None, required = 0):
        if libs is None:
            libs = []
        self.name = name
        self.wildcards = wildcards
        self.required = required
        self.paths = []
        self.path = None
        self.inc_dir = None
        self.lib_dir = None
        self.libs = libs
        self.found = False
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
            print ("Path for %s not found." % self.name)
            if self.required:
                print ('Too bad that is a requirement! Hand-fix the "Setup"')
        elif len(self.paths) == 1:
            self.path = self.paths[0]
            print ("Path for %s:%s" % (self.name, self.path))
        else:
            print ("Select path for %s:" % self.name)
            for i in range(len(self.paths)):
                print ("  %i=%s" % (i + 1, self.paths[i]))
            print ("  %i = <Nothing>" % 0)
            choice = raw_input("Select 0-%i (1=default):" % len(self.paths))
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
            self.found = True
            self.inc_dir = self.findhunt(self.path, Dependency.inc_hunt)
            self.lib_dir = self.findhunt(self.path, Dependency.lib_hunt)


class DependencyPython(object):
    def __init__(self, name, module, header):
        self.name = name
        self.lib_dir = ''
        self.inc_dir = ''
        self.libs = []
        self.cflags = ''
        self.found = False
        self.ver = '0'
        self.module = module
        self.header = header
 
    def configure(self):
        self.found = True
        if self.module:
            try:
                self.ver = __import__(self.module).__version__
            except ImportError:
                self.found = False
        if self.found and self.header:
            fullpath = os.path.join(get_python_inc(0), self.header)
            if not os.path.isfile(fullpath):
                found = 0
            else:
                self.inc_dir = os.path.split(fullpath)[0]
        if self.found:
            print ("%-8.8s: found %s" % (self.name, self.ver))
        else:
            print ("%-8.8s: not found" % self.name)


class DependencyDLL(Dependency):
    def __init__(self, dll_regex, lib=None, wildcards=None, libs=None, link=None):
        if lib is None:
            lib = link.libs[0]
        Dependency.__init__(self, 'COPYLIB_' + lib, wildcards, libs)
        self.lib_name = lib
        self.test = re.compile(dll_regex, re.I).match
        self.lib_dir = '_'
        self.found = True
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
                        print ("DLL for %s is %s" % (self.lib_name, self.lib_dir))
                        return
        print ("DLL for %s not found" % self.lib_name)

class DependencyWin(object):
    def __init__(self, name, cflags):
        self.name = name
        self.inc_dir = None
        self.lib_dir = None
        self.libs = []
        self.found = True
        self.cflags = cflags
        
    def configure(self):
        pass

class DependencyGroup(object):
    def __init__(self):
        self.dependencies =[]
        self.dlls = []

    def add(self, name, lib, wildcards, dll_regex, libs=None, required=0):
        if libs is None:
            libs = []
        dep = Dependency(name, wildcards, [lib], required)
        self.dependencies.append(dep)
        self.dlls.append(DependencyDLL(dll_regex, link=dep, libs=libs))

    def add_win(self, name, cflags):
        self.dependencies.append(DependencyWin(name, cflags))
                                 
    def add_dll(self, dll_regex, lib=None, wildcards=None, libs=None, link_lib=None):
        link = None
        if link_lib is not None:
            name = 'COPYLIB_' + link_lib
            for d in self.dlls:
                if d.name == name:
                    link = d
                    break
            else:
                raise KeyError("Link lib %s not found" % link_lib)
        self.dlls.append(DependencyDLL(dll_regex, lib, wildcards, libs, link))

    def configure(self):
        for d in self:
            d.configure()

    def __iter__(self):
        for d in self.dependencies:
            yield d
        for d in self.dlls:
            yield d

DEPS = DependencyGroup()
DEPS.add('SDL', 'SDL', ['SDL-[1-9].*'], r'(lib){0,1}SDL\.dll$', required=1)
DEPS.add('FONT', 'SDL_ttf', ['SDL_ttf-[2-9].*'], r'(lib){0,1}SDL_ttf\.dll$', ['SDL', 'z'])
DEPS.add('IMAGE', 'SDL_image', ['SDL_image-[1-9].*'], r'(lib){0,1}SDL_image\.dll$',
         ['SDL', 'jpeg', 'png', 'tiff'], 0),
DEPS.add('MIXER', 'SDL_mixer', ['SDL_mixer-[1-9].*'], r'(lib){0,1}SDL_mixer\.dll$',
         ['SDL', 'vorbisfile', 'smpeg'])
DEPS.add('SMPEG', 'smpeg', ['smpeg-[0-9].*', 'smpeg'], r'smpeg\.dll$', ['SDL'])
DEPS.add('PNG', 'png', ['libpng-[1-9].*'], r'(png|libpng13)\.dll$', ['z'])
DEPS.add('JPEG', 'jpeg', ['jpeg-[6-9]*'], r'(lib){0,1}jpeg\.dll$')
DEPS.add('PORTMIDI', 'portmidi', ['portmidi'], r'portmidi\.dll$')
#DEPS.add('FFMPEG', 'libavformat/avformat.h', 'libavformat.a', ['avformat', 'swscale', 'SDL_mixer'], r'avformat-52\.dll')   
dep = Dependency('FFMPEG', [r'avformat\.dll', r'swscale\.dll', r'SDL_mixer-[1-9].*'], ['avformat', 'swscale', 'SDL_mixer'], required=0)
DEPS.dependencies.append(dep)
DEPS.dlls.append(DependencyDLL(r'(lib){0,1}SDL_mixer\.dll$', link=dep, libs=['SDL', 'vorbisfile', 'smpeg']))
#DEPS.add('PORTTIME', 'porttime', ['porttime'], r'porttime\.dll$')
DEPS.add_dll(r'(lib){0,1}tiff\.dll$', 'tiff', ['tiff-[3-9].*'], ['jpeg', 'z'])
DEPS.add_dll(r'(z|zlib1)\.dll$', 'z', ['zlib-[1-9].*'])
DEPS.add_dll(r'(libvorbis-0|vorbis)\.dll$', 'vorbis', ['libvorbis-[1-9].*'],
             ['ogg'])
DEPS.add_dll(r'(libvorbisfile-3|vorbisfile)\.dll$', 'vorbisfile',
             link_lib='vorbis', libs=['vorbis'])
DEPS.add_dll(r'(libogg-0|ogg)\.dll$', 'ogg', ['libogg-[1-9].*'])
for d in get_definitions():
    DEPS.add_win(d.name, d.value)


def setup_prebuilt(prebuilt_dir):
    setup = open('Setup', 'w')
    try:
        try:
            setup_win_in = open(os.path.join(prebuilt_dir, 'Setup_Win.in'))
        except IOError:
            raise IOError("%s missing required Setup_Win.in" % prebuilt_dir)

        # Copy Setup.in to Setup, replacing the BeginConfig/EndConfig
        # block with prebuilt\Setup_Win.in .
        setup_in = open('Setup.in')
        try:
            do_copy = True
            for line in setup_in:
                if line.startswith('#--StartConfig'):
                    do_copy = False
                    setup.write(setup_win_in.read())
                    try:
                        setup_win_common_in = open('Setup_Win_Common.in')
                    except:
                        pass
                    else:
                        try:
                            setup.write(setup_win_common_in.read())
                        finally:
                            setup_win_common_in.close()
                elif line.startswith('#--EndConfig'):
                    do_copy = True
                elif do_copy:
                    setup.write(line)
        finally:
            setup_in.close()
    finally:
        setup.close()


def main():
    prebuilt_dir = 'prebuilt-' + as_machine_type(get_ptr_size())
    if os.path.isdir(prebuilt_dir):
        reply = raw_input('\nUse the SDL libraries in "%s"? [Y/n]' % prebuilt_dir)
        if not reply or reply[0].lower() != 'n':
            setup_prebuilt(prebuilt_dir)
            raise SystemExit()

    global DEPS
    
    DEPS.configure()
    return list(DEPS)

if __name__ == '__main__':
    print ("""This is the configuration subscript for Windows.
Please run "config.py" for full configuration.""")

