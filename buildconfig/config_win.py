"""Config on Windows"""

# **** The search part is broken. For instance, the png Visual Studio project
# places to dll in a directory not checked by this module.

try:
    from setup_win_common import get_definitions
except ImportError:
    from buildconfig.setup_win_common import get_definitions

import os, sys
import re
import logging
from glob import glob
from distutils.sysconfig import get_python_inc


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

class Dependency:
    huntpaths = ['..', '..\\..', '..\\*', '..\\..\\*']
    inc_hunt = ['include']
    lib_hunt = ['VisualC\\SDL\\Release', 'VisualC\\Release', 'Release', 'lib']
    check_hunt_roots = True
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
            self.find_lib = r"%s\.(a|lib)" % re.escape(libs[0])
        else:
            self.find_lib = find_lib
        self.libs = libs
        self.found = False
        self.cflags = ''
        self.prune_info = []
        self.fallback_inc = None
        self.fallback_lib = None

    def hunt(self):
        parent = os.path.abspath('..')
        for p in self.huntpaths:
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
            if self.fallback_inc and not self.inc_dir:
                self.inc_dir = self.fallback_inc[0]
            if self.fallback_lib and not self.lib_dir:
                self.lib_dir = self.fallback_lib[0]
                self.libs[0] = os.path.splitext(self.fallback_lib[2])[0]
            if self.inc_dir and self.lib_dir:
                if print_result:
                    print ("Path for %s found." % self.name)
                return True
            if print_result:
                print ("Path for %s not found." % self.name)
                for info in self.prune_info:
                    print(info)
                if self.required:
                    print ('Too bad that is a requirement! Hand-fix the "Setup"')
            return False
        elif len(self.paths) == 1:
            self.path = self.paths[0]
            if print_result:
                print ("Path for %s: %s" % (self.name, self.path))
        else:
            logging.warning("Multiple paths to choose from:%s", self.paths)
            self.path = self.paths[0]
            if print_result:
                print ("Path for %s: %s" % (self.name, self.path))
        return True

    def matchfile(self, path, match):
        try:
            entries = os.listdir(path)
        except OSError:
            pass
        else:
            for e in entries:
                if match(e) and os.path.isfile(os.path.join(path, e)):
                    return e

    def findhunt(self, base, paths, header_match=None, lib_match=None):
        for h in paths:
            hh = os.path.join(base, h)
            if header_match:
                header_file = self.matchfile(hh, header_match)
                if not header_file:
                    continue
            else:
                header_file = None
            if lib_match:
                lib_file = self.matchfile(hh, lib_match)
                if not lib_file:
                    continue
            else:
                lib_file = None
            if os.path.isdir(hh):
                return hh.replace('\\', '/'), header_file, lib_file

    def prunepaths(self):
        lib_match = re.compile(self.find_lib, re.I).match if self.find_lib else None
        header_match = re.compile(self.find_header, re.I).match if self.find_header else None
        prune = []
        for path in self.paths:
            inc_info = self.findhunt(path, Dependency.inc_hunt, header_match=header_match)
            lib_info = self.findhunt(path, Dependency.lib_hunt, lib_match=lib_match)
            if not inc_info or not lib_info:
                if inc_info:
                    self.prune_info.append('...Found include dir but no library dir in %s.' % (
                          path))
                    self.fallback_inc = inc_info
                if lib_info:
                    self.prune_info.append('...Found library dir but no include dir in %s.' % (
                          path))
                    self.fallback_lib = lib_info
                prune.append(path)
            else:
                self.inc_dir = inc_info[0]
                self.lib_dir = lib_info[0]
                self.libs[0] = os.path.splitext(lib_info[2])[0]
        self.paths = [p for p in self.paths if p not in prune]

    def configure(self):
        self.hunt()
        if self.check_hunt_roots:
            self.paths.extend(self.huntpaths)
        self.prunepaths()
        self.choosepath()
        if self.path:
            lib_match = re.compile(self.find_lib, re.I).match if self.find_lib else None
            header_match = re.compile(self.find_header, re.I).match if self.find_header else None
            inc_info = self.findhunt(self.path, Dependency.inc_hunt, header_match=header_match)
            lib_info = self.findhunt(self.path, Dependency.lib_hunt, lib_match=lib_match)
            if inc_info:
                self.inc_dir = inc_info[0]
            if lib_info:
                self.lib_info = lib_info[0]
                if lib_info[2]:
                    self.libs[0] = os.path.splitext(lib_info[2])[0]
        if self.lib_dir and self.inc_dir:
            print("...Library directory for %s: %s" % (self.name, self.lib_dir))
            print("...Include directory for %s: %s" % (self.name, self.inc_dir))
            self.found = True

class DependencyPython:
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
                self.found = False
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
        self.link = link

    def configure(self):
        if not self.path:
            if (self.link is None or not self.link.path) and self.wildcards:
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
        for p in self.huntpaths:
            if self.hunt_dll(self.lib_hunt, p):
                return True
        return False

    def hunt_dll(self, search_paths, root):
        for dir in search_paths:
            path = os.path.join(root, dir)
            try:
                entries = os.listdir(path)
            except OSError:
                pass
            else:
                for e in entries:
                    if self.test(e) and os.path.isfile(os.path.join(path, e)):
                        # Found
                        self.lib_dir = os.path.join(path, e).replace('\\', '/')
                        return True
        return False

class DependencyDummy:
    def __init__(self, name):
        self.name = name
        self.inc_dir = None
        self.lib_dir = None
        self.libs = []
        self.found = True
        self.cflags = ''

    def configure(self):
        pass

class DependencyWin:
    def __init__(self, name, cflags):
        self.name = name
        self.inc_dir = None
        self.lib_dir = None
        self.libs = []
        self.found = True
        self.cflags = cflags

    def configure(self):
        pass

class DependencyGroup:
    def __init__(self):
        self.dependencies =[]
        self.dlls = []

    def add(self, name, lib, wildcards, dll_regex, libs=None, required=0, find_header='', find_lib=''):
        if libs is None:
            libs = []
        if dll_regex:
            dep = Dependency(name, wildcards, [lib], required, find_header, find_lib)
            self.dependencies.append(dep)
            dll = DependencyDLL(dll_regex, link=dep, libs=libs)
            self.dlls.append(dll)
            dep.dll = dll
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

    def add_dummy(self, name):
        self.dependencies.append(DependencyDummy(name))

    def find(self, name):
        for dep in self:
            if dep.name == name:
                return dep

    def configure(self):
        for d in self.dependencies:
            if not getattr(d, '_configured', False):
                d.configure()
                d._configured = True
        for d in self.dlls:
            if not getattr(d, '_configured', False):
                d.configure()
                d._configured = True

                # create a lib
                if d.found and d.link and not d.link.lib_dir:
                    try:
                        from . import vstools
                    except ImportError:
                        from buildconfig import vstools
                    from os.path import splitext
                    nonext_name = splitext(d.lib_dir)[0]
                    def_file = '%s.def' % nonext_name
                    basename = os.path.basename(nonext_name)
                    print('Building lib from %s: %s.lib...' % (
                        os.path.basename(d.lib_dir),
                        basename
                    ))
                    vstools.dump_def(d.lib_dir, def_file=def_file)
                    vstools.lib_from_def(def_file)
                    d.link.lib_dir = os.path.dirname(d.lib_dir)
                    d.link.libs[0] = basename
                    d.link.configure()

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
    DEPS.add_dll(r'(lib){0,1}tiff[-0-9]*\.dll$', 'tiff', ['tiff-[0-9]*'], ['jpeg', 'z'])
    DEPS.add_dll(r'(z|zlib1)\.dll$', 'z', ['zlib-[1-9].*'])
    DEPS.add_dll(r'(lib)?webp[-0-9]*\.dll$', 'webp', ['*webp-[0-9]*'])

def setup():
    DEPS = DependencyGroup()

    DEPS.add('SDL', 'SDL2', ['SDL2-[1-9].*'], r'(lib){0,1}SDL2\.dll$', required=1)
    DEPS.add('PORTMIDI', 'portmidi', ['portmidi'], r'portmidi\.dll$', find_header=r'portmidi\.h')
    #DEPS.add('PORTTIME', 'porttime', ['porttime'], r'porttime\.dll$')
    DEPS.add_dummy('PORTTIME')
    DEPS.add('MIXER', 'SDL2_mixer', ['SDL2_mixer-[1-9].*'], r'(lib){0,1}SDL2_mixer\.dll$',
             ['SDL', 'vorbisfile'])
    DEPS.add('PNG', 'png', ['SDL2_image-[2-9].*', 'libpng-[1-9].*'], r'(png|libpng)[-0-9]*\.dll$', ['z'],
             find_header=r'png\.h', find_lib=r'(lib)?png1[-0-9]*\.lib')
    DEPS.add('JPEG', 'jpeg', ['SDL2_image-[2-9].*', 'jpeg-9*'], r'(lib){0,1}jpeg-9\.dll$',
             find_header=r'jpeglib\.h', find_lib=r'(lib)?jpeg-9\.lib')
    DEPS.add('IMAGE', 'SDL2_image', ['SDL2_image-[1-9].*'], r'(lib){0,1}SDL2_image\.dll$',
             ['SDL', 'jpeg', 'png', 'tiff'], 0)
    DEPS.add('FONT', 'SDL2_ttf', ['SDL2_ttf-[2-9].*'], r'(lib){0,1}SDL2_ttf\.dll$', ['SDL', 'z', 'freetype'])
    DEPS.add('FREETYPE', 'freetype', ['freetype-[1-9].*'], r'(lib){0,1}freetype[-0-9]*\.dll$',
             find_header=r'ft2build\.h', find_lib=r'(lib)?freetype[-0-9]*\.lib')
    DEPS.configure()
    _add_sdl2_dll_deps(DEPS)
    for d in get_definitions():
        DEPS.add_win(d.name, d.value)
    DEPS.configure()

    return list(DEPS)

def setup_prebuilt_sdl2(prebuilt_dir):
    Dependency.huntpaths[:] = [prebuilt_dir]
    Dependency.lib_hunt.extend([
        '',
        'lib',
        os.path.join('lib', get_machine_type()),
    ])
    Dependency.inc_hunt.append('')

    DEPS = DependencyGroup()

    DEPS.add('SDL', 'SDL2', ['SDL2-[1-9].*'], r'(lib){0,1}SDL2\.dll$', required=1)
    fontDep = DEPS.add('FONT', 'SDL2_ttf', ['SDL2_ttf-[2-9].*'], r'(lib){0,1}SDL2_ttf\.dll$', ['SDL', 'z', 'freetype'])
    imageDep = DEPS.add('IMAGE', 'SDL2_image', ['SDL2_image-[1-9].*'], r'(lib){0,1}SDL2_image\.dll$',
                        ['SDL', 'jpeg', 'png', 'tiff'], 0)
    mixerDep = DEPS.add('MIXER', 'SDL2_mixer', ['SDL2_mixer-[1-9].*'], r'(lib){0,1}SDL2_mixer\.dll$',
                        ['SDL', 'vorbisfile'])
    DEPS.add('PORTMIDI', 'portmidi', ['portmidi'], r'portmidi\.dll$', find_header=r'portmidi\.h')
    #DEPS.add('PORTTIME', 'porttime', ['porttime'], r'porttime\.dll$')
    DEPS.add_dummy('PORTTIME')
    DEPS.configure()

    # force use of the correct freetype DLL
    ftDep = DEPS.add('FREETYPE', 'freetype', ['SDL2_ttf-[2-9].*', 'freetype-[1-9].*'], r'(lib)?freetype[-0-9]*\.dll$',
                     find_header=r'ft2build\.h', find_lib=r'libfreetype[-0-9]*\.lib')
    ftDep.path = fontDep.path
    ftDep.inc_dir = [
        os.path.join(prebuilt_dir, 'include').replace('\\', '/')
    ]
    ftDep.inc_dir.append('%s/freetype2' % ftDep.inc_dir[0])
    ftDep.found = True

    png = DEPS.add('PNG', 'png', ['SDL2_image-[2-9].*', 'libpng-[1-9].*'], r'(png|libpng)[-0-9]*\.dll$', ['z'],
                   find_header=r'png\.h', find_lib=r'(lib)?png1[-0-9]*\.lib')
    png.path = imageDep.path
    png.inc_dir = [os.path.join(prebuilt_dir, 'include').replace('\\', '/')]
    png.found = True
    jpeg = DEPS.add('JPEG', 'jpeg', ['SDL2_image-[2-9].*', 'jpeg-9*'], r'(lib){0,1}jpeg-9\.dll$',
                   find_header=r'jpeglib\.h', find_lib=r'(lib)?jpeg-9\.lib')
    jpeg.path = imageDep.path
    jpeg.inc_dir = [os.path.join(prebuilt_dir, 'include').replace('\\', '/')]
    jpeg.found = True

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
        if dllPaths.get(dll.lib_name):
            dll.path = dllPaths.get(dll.lib_name)

    for d in get_definitions():
        DEPS.add_win(d.name, d.value)

    DEPS.configure()
    return list(DEPS)

def main():
    machine_type = get_machine_type()
    prebuilt_dir = 'prebuilt-%s' % machine_type
    use_prebuilt = '-prebuilt' in sys.argv

    auto_download = 'PYGAME_DOWNLOAD_PREBUILT' in os.environ
    if auto_download:
        auto_download = os.environ['PYGAME_DOWNLOAD_PREBUILT'] == '1'

    try:
        from . import download_win_prebuilt
    except ImportError:
        import download_win_prebuilt

    download_kwargs = {
        'x86': False,
        'x64': False,
    }
    download_kwargs[machine_type] = True

    if not auto_download:
        if (not download_win_prebuilt.cached(**download_kwargs) or\
            not os.path.isdir(prebuilt_dir))\
            and download_win_prebuilt.ask(**download_kwargs):
            use_prebuilt = True
    else:
        download_win_prebuilt.update(**download_kwargs)

    if os.path.isdir(prebuilt_dir):
        if not use_prebuilt:
            if 'PYGAME_USE_PREBUILT' in os.environ:
                use_prebuilt = os.environ['PYGAME_USE_PREBUILT'] == '1'
            else:
                logging.warning('Using the SDL libraries in "%s".' % prebuilt_dir)
                use_prebuilt = True

        if use_prebuilt:
            return setup_prebuilt_sdl2(prebuilt_dir)
    else:
        print ("Note: cannot find directory \"%s\"; do not use prebuilts." % prebuilt_dir)
    return setup()

if __name__ == '__main__':
    print ("""This is the configuration subscript for Windows.
Please run "config.py" for full configuration.""")

    import sys
    if "--download" in sys.argv:
        try:
            from . import download_win_prebuilt
        except ImportError:
            import download_win_prebuilt
        download_win_prebuilt.ask()
