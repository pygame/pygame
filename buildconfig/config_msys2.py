"""Config on MSYS2"""

# The search logic is adapted from config_win.
# This assumes that the PyGame dependencies are resolved
# by MSYS2 packages.

try:
    from setup_win_common import get_definitions
except ImportError:
    from buildconfig.setup_win_common import get_definitions

import os
import sys
import re
import logging
import subprocess
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
    raise ValueError("Unknown pointer size {}".format(size))

def get_machine_type():
    return as_machine_type(get_ptr_size())

def get_absolute_win_path(msys2_path):
    output = subprocess.run(['cygpath', '-w', msys2_path],
                            capture_output=True, text=True)
    if output.returncode != 0:
        raise Exception(f"Could not get absolute Windows path: {msys2_path}")
    else:
        return output.stdout.strip()

class Dependency:
    huntpaths = ['..', '../..', '../*', '../../*']
    inc_hunt = ['include']
    lib_hunt = ['lib']
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
            # Windows dllimport libs (.lib) are called .dll.a in MSYS2
            self.find_lib = r"lib%s\.dll\.a" % re.escape(libs[0])
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
            # for w in self.wildcards:
            # found = glob(os.path.join(p, w))
            found = glob(p)
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
                # MSYS2: lib<name>.dll.a -> <name>
                self.libs[0] = os.path.splitext(self.fallback_lib[2])[0].lstrip('lib').rstrip('.dll')
            if self.inc_dir and self.lib_dir:
                if print_result:
                    print(f"Path for {self.name} found.")
                return True
            if print_result:
                print(f"Path for {self.name} not found.")
                for info in self.prune_info:
                    print(info)
                if self.required:
                    print('Too bad that is a requirement! Hand-fix the "Setup"')
            return False
        elif len(self.paths) == 1:
            self.path = self.paths[0]
            if print_result:
                print(f"Path for {self.name}: {self.path}")
        else:
            logging.warning("Multiple paths to choose from:%s", self.paths)
            self.path = self.paths[0]
            if print_result:
                print(f"Path for {self.name}: {self.path}")
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
                # MSYS2: lib<name>.dll.a -> <name>
                self.libs[0] = os.path.splitext(lib_info[2])[0].lstrip('lib').rstrip('.dll')
        self.paths = [p for p in self.paths if p not in prune]

    def configure(self):
        self.hunt()
        # In config MSYS2, huntpaths always includes a root
        # if self.check_hunt_roots:
        #     self.paths.extend(self.huntpaths)
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
                    # MSYS2: lib<name>.dll.a -> <name>
                    self.libs[0] = os.path.splitext(lib_info[2])[0].lstrip('lib').rstrip('.dll')
        if self.lib_dir and self.inc_dir:
            print(f"...Library directory for {self.name}: {self.lib_dir}")
            print(f"...Include directory for {self.name}: {self.inc_dir}")
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
            print("%-8.8s: found %s" % (self.name, self.ver))
        else:
            print("%-8.8s: not found" % self.name)

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
            print(f"DLL for {self.lib_name}: {self.lib_dir}")
            self.found = True
        else:
            print(f"No DLL for {self.lib_name}: not found!")
            if self.required:
                print('Too bad that is a requirement! Hand-fix the "Setup"')

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
                raise KeyError(f"Link lib {link_lib} not found")
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
                    def_file = f'{nonext_name}.def'
                    basename = os.path.basename(nonext_name)
                    print(f'Building lib from {os.path.basename(d.lib_dir)}: {basename}.lib...')
                    vstools.dump_def(d.lib_dir, def_file=def_file)
                    vstools.lib_from_def(def_file)
                    d.link.lib_dir = os.path.dirname(d.lib_dir)
                    d.link.libs[0] = basename
                    d.link.configure()

    def __iter__(self):
        yield from self.dependencies
        yield from self.dlls

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


def setup_prebuilt_sdl2(prebuilt_dir):
    Dependency.huntpaths[:] = [prebuilt_dir]
    Dependency.lib_hunt.extend([
        '',
        # MSYS2 installs prebuilt .dll in /mingw*/bin
        'bin',
        'lib',
    ])
    Dependency.inc_hunt.append('')

    DEPS = DependencyGroup()

    sdlDep = DEPS.add('SDL', 'SDL2', ['SDL2-[1-9].*'], r'(lib){0,1}SDL2\.dll$', find_header=r'SDL\.h', required=1)
    sdlDep.inc_dir = [
        os.path.join(prebuilt_dir, 'include').replace('\\', '/')
    ]
    sdlDep.inc_dir.append(f'{sdlDep.inc_dir[0]}/SDL2')
    fontDep = DEPS.add('FONT', 'SDL2_ttf', ['SDL2_ttf-[2-9].*'], r'(lib){0,1}SDL2_ttf\.dll$', ['SDL', 'z', 'freetype'])
    imageDep = DEPS.add('IMAGE', 'SDL2_image', ['SDL2_image-[1-9].*'], r'(lib){0,1}SDL2_image\.dll$',
                        ['SDL', 'jpeg', 'png', 'tiff'], 0)
    mixerDep = DEPS.add('MIXER', 'SDL2_mixer', ['SDL2_mixer-[1-9].*'], r'(lib){0,1}SDL2_mixer\.dll$',
                        ['SDL', 'vorbisfile'])
    DEPS.add('PORTMIDI', 'portmidi', ['portmidi'], r'(lib){0,1}portmidi\.dll$', find_header=r'portmidi\.h')
    # #DEPS.add('PORTTIME', 'porttime', ['porttime'], r'porttime\.dll$')
    DEPS.add_dummy('PORTTIME')

    # force use of the correct freetype DLL
    ftDep = DEPS.add('FREETYPE', 'freetype', ['SDL2_ttf-[2-9].*', 'freetype-[1-9].*'], r'(lib)?freetype[-0-9]*\.dll$',
                     find_header=r'ft2build\.h', find_lib=r'libfreetype[-0-9]*\.dll\.a')
    ftDep.path = fontDep.path
    ftDep.inc_dir = [
        os.path.join(prebuilt_dir, 'include').replace('\\', '/')
    ]
    ftDep.inc_dir.append(f'{ftDep.inc_dir[0]}/freetype2')
    ftDep.found = True

    png = DEPS.add('PNG', 'png', ['SDL2_image-[2-9].*', 'libpng-[1-9].*'], r'(png|libpng)[-0-9]*\.dll$', ['z'],
                   find_header=r'png\.h', find_lib=r'(lib)?png1[-0-9]*\.dll\.a')
    png.path = imageDep.path
    png.inc_dir = [os.path.join(prebuilt_dir, 'include').replace('\\', '/')]
    png.found = True
    jpeg = DEPS.add('JPEG', 'jpeg', ['SDL2_image-[2-9].*', 'jpeg(-8*)?'], r'(lib){0,1}jpeg-8\.dll$',
                   find_header=r'jpeglib\.h', find_lib=r'(lib)?jpeg(-8)?\.dll\.a')
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


def main(auto_config=False):
    machine_type = get_machine_type()

    # config MSYS2 always requires prebuilt dependencies, in the
    # form of packages available in MSYS2.
    download_prebuilt = 'PYGAME_DOWNLOAD_PREBUILT' in os.environ
    if download_prebuilt:
        download_prebuilt = os.environ['PYGAME_DOWNLOAD_PREBUILT'] == '1'
    else:
        download_prebuilt = True

    try:
        from . import download_msys2_prebuilt
    except ImportError:
        import download_msys2_prebuilt

    download_kwargs = {
        'x86': False,
        'x64': False,
    }
    download_kwargs[machine_type] = True
    if download_prebuilt:
        download_msys2_prebuilt.update(**download_kwargs)

    # MSYS2 config only supports setup with prebuilt dependencies
    # The prebuilt dir is the MinGW root from the MSYS2
    # installation path. Since we're currently running in a native
    # binary, this Python has no notion of MSYS2 or MinGW paths, so
    # we convert the prebuilt dir to a Windows absolute path.
    # e.g. /mingw64 (MSYS2)  ->  C:/msys64/mingw64 (Windows)
    prebuilt_msys_dir = {
        'x86': '/mingw32',
        'x64': '/mingw64'
    }
    prebuilt_dir = get_absolute_win_path(prebuilt_msys_dir[machine_type])
    return setup_prebuilt_sdl2(prebuilt_dir)


if __name__ == '__main__':
    print("""This is the configuration subscript for MSYS2.
Please run "config.py" for full configuration.""")
    if "--download" in sys.argv:
        try:
            from . import download_msys2_prebuilt
        except ImportError:
            import download_msys2_prebuilt
        download_msys2_prebuilt.update()
