"""Config on Windows"""

# **** The search part is broken. For instance, the png Visual Studio project
# places to dll in a directory not checked by this module.

try:
    from setup_win_common import get_definitions
except:
    from buildconfig.setup_win_common import get_definitions

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

def get_machine_type():
    return as_machine_type(get_ptr_size())

class Dependency(object):
    inc_hunt = ['include']
    lib_hunt = ['VisualC\\SDL\\Release', 'VisualC\\Release', 'Release', 'lib']
    def __init__(self, name, wildcards, libs=None, required=0, find_header='', find_lib=''):
        if libs is None:
            libs = []
        self.name = name
        self.wildcards = wildcards
        self.required = required
        self.paths = []
        self.path = None
        self.inc_dir = None
        self.lib_dir = None
        self.find_header = find_header
        if not find_lib and libs:
            self.find_lib = "%s\.(a|lib)" % re.escape(libs[0])
        else:
            self.find_lib = find_lib
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

    def choosepath(self, print_result=True):
        if not self.paths:
            if print_result:
                print ("Path for %s not found." % self.name)
                if self.required:
                    print ('Too bad that is a requirement! Hand-fix the "Setup"')
            return False
        elif len(self.paths) == 1:
            self.path = self.paths[0]
            if print_result:
                print ("Path for %s: %s" % (self.name, self.path))
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
        return True

    def matchfile(self, path, match):
        try:
            entries = os.listdir(path)
        except:
            pass
        else:
            for e in entries:
                if match(e) and os.path.isfile(os.path.join(path, e)):
                    return e

    def findhunt(self, base, paths, header_match=None, lib_match=None):
        for h in paths:
            hh = os.path.join(base, h)
            if header_match and not self.matchfile(hh, header_match):
                continue
            if lib_match and not self.matchfile(hh, lib_match):
                continue
            if os.path.isdir(hh):
                return hh.replace('\\', '/')
        if header_match:
            print("...Header(s) for %s could not be found!" % self.name)
        if lib_match:
            print("...Library for %s could not be found!" % self.name)

    def configure(self):
        self.hunt()
        self.choosepath()
        if self.path:
            lib_match = re.compile(self.find_lib, re.I).match if self.find_lib else None
            header_match = re.compile(self.find_header, re.I).match if self.find_header else None
            self.inc_dir = self.findhunt(self.path, Dependency.inc_hunt, header_match=header_match)
            self.lib_dir = self.findhunt(self.path, Dependency.lib_hunt, lib_match=lib_match)
            print("...Library directory for %s: %s" % (self.name, self.lib_dir))
            print("...Include directory for %s: %s" % (self.name, self.inc_dir))
            if self.inc_dir or self.lib_dir:
                self.found = True


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
    check_hunt_roots = True

    def __init__(self, dll_regex, lib=None, wildcards=None, libs=None, link=None):
        if lib is None:
            lib = link.libs[0]
        Dependency.__init__(self, 'COPYLIB_' + lib, wildcards, libs)
        self.lib_name = lib
        self.test = re.compile(dll_regex, re.I).match
        self.lib_dir = '_'
        self.link = link

    def configure(self):
        if not self.path:
            if self.link is None and self.wildcards:
                self.hunt()
                self.choosepath(print_result=False)
            else:
                self.path = self.link.path
        if self.path is not None:
            self.hunt_dll(self.lib_hunt, self.path)
        elif self.check_hunt_roots:
            self.check_roots()

        if self.lib_dir != '_':
            print ("DLL for %s: %s" % (self.lib_name, self.lib_dir))
            self.found = True
        else:
            print ("No DLL for %s: not found!" % (self.lib_name))
            if self.required:
                print ('Too bad that is a requirement! Hand-fix the "Setup"')

    def check_roots(self):
        parent = os.path.abspath('..')
        for p in huntpaths:
            if self.hunt_dll(self.lib_hunt, p):
                return True
        return False

    def hunt_dll(self, search_paths, root):
        for dir in search_paths:
            path = os.path.join(root, dir)
            try:
                entries = os.listdir(path)
            except:
                pass
            else:
                for e in entries:
                    if self.test(e) and os.path.isfile(os.path.join(path, e)):
                        # Found
                        self.lib_dir = os.path.join(path, e).replace('\\', '/')
                        return True
        return False

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

    def add(self, name, lib, wildcards, dll_regex, libs=None, required=0, find_header='', find_lib=''):
        if libs is None:
            libs = []
        if dll_regex:
            dep = Dependency(name, wildcards, [lib], required, find_header, find_lib)
            self.dependencies.append(dep)
            self.dlls.append(DependencyDLL(dll_regex, link=dep, libs=libs))
        else:
            dep = Dependency(name, wildcards, [lib] + libs, required, find_header, find_lib)
            self.dependencies.append(dep)
        return dep

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
        dep = DependencyDLL(dll_regex, lib, wildcards, libs, link)
        self.dlls.append(dep)
        return dep

    def find(self, name):
        for dep in self:
            if dep.name == name:
                return dep

    def configure(self):
        for d in self:
            if not getattr(d, '_configured', False):
                d.configure()
                d._configured = True

    def __iter__(self):
        for d in self.dependencies:
            yield d
        for d in self.dlls:
            yield d

def _add_sdl2_dll_deps(DEPS):
    # MIXER
    DEPS.add_dll(r'(libvorbis-0|vorbis)\.dll$', 'vorbis', ['libvorbis-[1-9].*'],
                 ['ogg'])
    DEPS.add_dll(r'(libvorbisfile-3|vorbisfile)\.dll$', 'vorbisfile',
                 link_lib='vorbis', libs=['vorbis'])
    DEPS.add_dll(r'(libogg-0|ogg)\.dll$', 'ogg', ['libogg-[1-9].*'])
    DEPS.add_dll(r'(lib)?FLAC[-0-9]*\.dll$', 'flac', ['*FLAC-[0-9]*'])
    DEPS.add_dll(r'(lib)?modplug[-0-9]*\.dll$', 'modplug', ['*modplug-[0-9]*'])
    DEPS.add_dll(r'(lib)?mpg123[-0-9]*\.dll$', 'mpg123', ['*mpg123-[0-9]*'])
    DEPS.add_dll(r'(lib)?opus[-0-9]*\.dll$', 'opus', ['*opus-[0-9]*'])
    DEPS.add_dll(r'(lib)?opusfile[-0-9]*\.dll$', 'opusfile', ['*opusfile-[0-9]*'])
    # IMAGE
    DEPS.add_dll(r'(png|libpng.*)\.dll$', 'png', ['libpng-[1-9].*'], ['z'])
    DEPS.add_dll(r'(lib){0,1}jpeg[-0-9]*\.dll$', 'jpeg', ['jpeg-[6-9]*'])
    DEPS.add_dll(r'(lib){0,1}tiff[-0-9]*\.dll$', 'tiff', ['tiff-[0-9]*'], ['jpeg', 'z'])
    DEPS.add_dll(r'(z|zlib1)\.dll$', 'z', ['zlib-[1-9].*'])
    DEPS.add_dll(r'(lib)?webp[-0-9]*\.dll$', 'webp', ['*webp-[0-9]*'])
    # TTF
    DEPS.add_dll(r'(lib)?freetype[-0-9]*\.dll$', 'freetype', ['(lib)?freetype[-0-9]*\.dll*'])

def setup(sdl2):
    DEPS = DependencyGroup()

    if not sdl2:
        DEPS.add('SDL', 'SDL', ['SDL-[1-9].*'], r'(lib){0,1}SDL\.dll$', required=1)
        DEPS.add('FONT', 'SDL_ttf', ['SDL_ttf-[2-9].*'], r'(lib){0,1}SDL_ttf\.dll$', ['SDL', 'z'])
        DEPS.add('IMAGE', 'SDL_image', ['SDL_image-[1-9].*'], r'(lib){0,1}SDL_image\.dll$',
                 ['SDL', 'jpeg', 'png', 'tiff'], 0),
        DEPS.add('MIXER', 'SDL_mixer', ['SDL_mixer-[1-9].*'], r'(lib){0,1}SDL_mixer\.dll$',
                 ['SDL', 'vorbisfile'])
        DEPS.add('PNG', 'png', ['libpng-[1-9].*'], r'(png|libpng13)\.dll$', ['z'])
        DEPS.add('JPEG', 'jpeg', ['jpeg-[6-9]*'], r'(lib){0,1}jpeg\.dll$')
        DEPS.add('PORTMIDI', 'portmidi', ['portmidi'], r'portmidi\.dll$')
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
    else:
        DEPS.add('SDL', 'SDL2', ['SDL2-[1-9].*'], r'(lib){0,1}SDL2\.dll$', required=1)
        DEPS.add('PORTMIDI', 'portmidi', ['portmidi'], r'portmidi\.dll$')
        #DEPS.add('PORTTIME', 'porttime', ['porttime'], r'porttime\.dll$')
        DEPS.add('MIXER', 'SDL2_mixer', ['SDL2_mixer-[1-9].*'], r'(lib){0,1}SDL2_mixer\.dll$',
                 ['SDL', 'vorbisfile'])
        DEPS.add('IMAGE', 'SDL2_image', ['SDL2_image-[1-9].*'], r'(lib){0,1}SDL2_image\.dll$',
                 ['SDL', 'jpeg', 'png', 'tiff'], 0)
        DEPS.add('FONT', 'SDL2_ttf', ['SDL2_ttf-[2-9].*'], r'(lib){0,1}SDL2_ttf\.dll$', ['SDL', 'z'])
        _add_sdl2_dll_deps(DEPS)
        for d in get_definitions():
            DEPS.add_win(d.name, d.value)

    DEPS.configure()
    return list(DEPS)

def setup_prebuilt_sdl2(prebuilt_dir):
    huntpaths[:] = [prebuilt_dir]
    Dependency.lib_hunt.extend([
        '',
        os.path.join('lib', get_machine_type()),
    ])

    DEPS = DependencyGroup()

    DEPS.add('SDL', 'SDL2', ['SDL2-[1-9].*'], r'(lib){0,1}SDL2\.dll$', required=1)
    fontDep = DEPS.add('FONT', 'SDL2_ttf', ['SDL2_ttf-[2-9].*'], r'(lib){0,1}SDL2_ttf\.dll$', ['SDL', 'z'])
    imageDep = DEPS.add('IMAGE', 'SDL2_image', ['SDL2_image-[1-9].*'], r'(lib){0,1}SDL2_image\.dll$',
                        ['SDL', 'jpeg', 'png', 'tiff'], 0)
    mixerDep = DEPS.add('MIXER', 'SDL2_mixer', ['SDL2_mixer-[1-9].*'], r'(lib){0,1}SDL2_mixer\.dll$',
                        ['SDL', 'vorbisfile'])
    DEPS.add('PORTMIDI', 'portmidi', ['portmidi'], r'portmidi\.dll$')
    #DEPS.add('PORTTIME', 'porttime', ['porttime'], r'porttime\.dll$')

    DEPS.configure()

    dllPaths = {
        'png': imageDep.path,
        'jpeg': imageDep.path,
        'tiff': imageDep.path,
        'z': imageDep.path,
        'webp': imageDep.path,

        'vorbis': mixerDep.path,
        'vorbisfile': mixerDep.path,
        'ogg': mixerDep.path,
        'flac': mixerDep.path,
        'modplug': mixerDep.path,
        'mpg123': mixerDep.path,
        'opus': mixerDep.path,
        'opusfile': mixerDep.path,

        'freetype': fontDep.path,
    }
    _add_sdl2_dll_deps(DEPS)
    for dll in DEPS.dlls:
        dll.path = dllPaths.get(dll.lib_name)

    for d in get_definitions():
        DEPS.add_win(d.name, d.value)

    DEPS.configure()
    return list(DEPS)

def setup_prebuilt_sdl1(prebuilt_dir):
    setup_ = open('Setup', 'w')
    is_pypy = '__pypy__' in sys.builtin_module_names
    import platform
    is_python3 = platform.python_version().startswith('3')

    try:
        try:
            setup_win_in = open(os.path.join(prebuilt_dir, 'Setup_Win.in'))
        except IOError:
            raise IOError("%s missing required Setup_Win.in" % prebuilt_dir)

        # Copy Setup.in to Setup, replacing the BeginConfig/EndConfig
        # block with prebuilt\Setup_Win.in .
        setup_in = open(os.path.join('buildconfig', 'Setup.SDL1.in'))
        try:
            do_copy = True
            for line in setup_in:
                if is_pypy and is_python3:
                    if line.startswith('_freetype'):
                        continue
                if line.startswith('#--StartConfig'):
                    do_copy = False
                    setup_.write(setup_win_in.read())
                    try:
                        setup_win_common_in = open(os.path.join('buildconfig', 'Setup_Win_Common.in'))
                    except:
                        pass
                    else:
                        try:
                            setup_.write(setup_win_common_in.read())
                        finally:
                            setup_win_common_in.close()
                elif line.startswith('#--EndConfig'):
                    do_copy = True
                elif do_copy:
                    setup_.write(line)
        finally:
            setup_in.close()
    finally:
        setup_.close()


def main(sdl2=False):
    prebuilt_dir = 'prebuilt-' + get_machine_type()
    if os.path.isdir(prebuilt_dir):
        use_prebuilt = '-prebuilt' in sys.argv
        if not use_prebuilt:
            if 'PYGAME_USE_PREBUILT' in os.environ:
                use_prebuilt = os.environ['PYGAME_USE_PREBUILT'] == '1'
            else:
                reply = raw_input('\nUse the SDL libraries in "%s"? [Y/n]' % prebuilt_dir)
                use_prebuilt = (not reply) or reply[0].lower() != 'n'

        if use_prebuilt:
            if sdl2:
                return setup_prebuilt_sdl2(prebuilt_dir)
            setup_prebuilt_sdl1(prebuilt_dir)
            raise SystemExit()
    else:
        print ("Note: cannot find directory \"%s\"; do not use prebuilts." % prebuilt_dir)
    return setup(sdl2)

if __name__ == '__main__':
    print ("""This is the configuration subscript for Windows.
Please run "config.py" for full configuration.""")
