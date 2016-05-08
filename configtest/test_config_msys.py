# program test_config_msys.py

"""Test config_msys.py against a dummy directory structure.

This test requires MSYS. It is specific to the Pygame 1.9.0 dependencies
as built by msys_build_deps.py.
"""

import os
import os.path
import sys

test_dir = './testdir'
if not os.path.isdir(test_dir):
    print "Test directory %s not found." % test_dir

os.environ['LOCALBASE'] = test_dir
sys.path.append('..')

import dll
import config_msys
import msys

import unittest

cwd = os.getcwd()
m = msys.Msys(require_mingw=False)
os.environ['SDL_CONFIG'] = m.windows_to_msys(os.path.join(cwd, 'test-sdl-config'))
dependencies = dict([(dep.name, dep) for dep in config_msys.main()])
del m

class Dependency(object):
    # Holds dependency info
    def __init__(self, name=None, inc_dir_rel=None, lib_dir_rel=None, libs=None, cflags=None):
        if libs is None:
            if name is None:
                libs = []
            else:
                libs = [dll.name_to_root(name)]
        if cflags is None:
            cflags = ''
        self.libs = libs
        self.inc_dir = None
        self.lib_dir = None
        if inc_dir_rel is not None:
            self.inc_dir = '%s/%s' % (test_dir, inc_dir_rel)
        if lib_dir_rel is not None:
            self.lib_dir = '%s/%s' % (test_dir, lib_dir_rel)
        self.cflags = cflags

class DependencyDLL(Dependency):
    def __init__(self, name=None, inc_dir_rel=None, lib_dir_rel=None, libs=None):
        if libs is None:
            if name is not None:
                libs = dll.libraries(name)
        super(DependencyDLL, self).__init__(name, inc_dir_rel, lib_dir_rel, libs)

class RunConfigTestCase(unittest.TestCase):
    """Test dependencies returned by config_msys.main()"""

    # Pygame dependencies
    expectations = {
        'SDL': Dependency('SDL', 'include/sdl', 'lib',
                          cflags='-I./SDL/include -DSDL_MACRO=1 '
                                 '-Xlinker -Wl,sdl_1,sdl_2 '
                                 '-L./SDL/lib -lSDL '),  # uses test-sdl-config script
        'FONT': Dependency('FONT', 'include/sdl', 'lib'),
        'IMAGE': Dependency('IMAGE', 'include/sdl', 'lib'),
        'MIXER': Dependency('MIXER', 'include', 'lib'),  # A deviant include dir
        'PNG': Dependency('PNG', 'include/libpng12', 'lib'),
        'JPEG': Dependency('JPEG', 'include/sdl', 'lib'),  # A deviant include dir
        'SCRAP': Dependency(cflags='-luser32 -lgdi32'),
        'COPYLIB_SDL': DependencyDLL('SDL', lib_dir_rel='bin/sdl.dll'),
        'COPYLIB_SDL_ttf': DependencyDLL('FONT', lib_dir_rel='bin/sdl_ttf.dll'),  # Where DLLs likely are
        'COPYLIB_SDL_image': DependencyDLL('IMAGE', lib_dir_rel='bin/sdl_image.dll'),
        'COPYLIB_SDL_mixer': DependencyDLL('MIXER', lib_dir_rel='lib/sdl_mixer.dll'),  # Where the search starts
        'COPYLIB_tiff': DependencyDLL('TIFF', lib_dir_rel='bin/libtiff.dll'),
        'COPYLIB_png12': DependencyDLL('PNG', lib_dir_rel='bin/libpng12-0.dll'),
        'COPYLIB_jpeg': DependencyDLL('JPEG', lib_dir_rel='bin/jpeg.dll'),
        'COPYLIB_z': DependencyDLL('Z', lib_dir_rel='bin/zlib1.dll'),
        'COPYLIB_vorbisfile': DependencyDLL('VORBISFILE', lib_dir_rel='bin/libvorbisfile-3.dll'),
        'COPYLIB_vorbis': DependencyDLL('VORBIS', lib_dir_rel='bin/libvorbis-0.dll'),
        'COPYLIB_ogg': DependencyDLL('OGG', lib_dir_rel='bin/libogg-0.dll'),
        'COPYLIB_freetype': DependencyDLL('FREETYPE', lib_dir_rel='bin/libfreetype-6.dll'),
        }

    def test_dependencies(self):
        """Ensure all dependencies are present"""
        self.failUnlessEqual(len(dependencies), len(self.expectations))
        for name in self.expectations:
            self.failUnless(name in dependencies, name)

    def test_dll_match(self):
        """Ensure DLLs match with dll.py."""
        for name in dll.regexs:
            self.failUnless('COPYLIB_' + dll.name_to_root(name) in dependencies, name)

    def test_found(self):
        """Ensure all dependencies were found"""
        for dep in dependencies.values():
            self.failUnless(dep.found, dep.name)

    # def test_not_found(self):
    # No easy way to test the case where something is missing

    def test_libs(self):
        """Ensure each dependency has the proper libraries"""
        from config_msys import DependencyProg
        
        for name, dep in dependencies.items():
            if isinstance(dep, DependencyProg):
                # Do not know how to test this one.
                continue
            dlibs = dep.libs
            elibs = self.expectations[name].libs
            self.failUnlessEqual(dlibs, elibs, "%s: %s != %s" % (name, dlibs, elibs))

    def test_proper_include_paths(self):
        """Ensure each dependency has found its include directory"""
        from config_msys import DependencyProg
        
        for name, dep in dependencies.items():
            if isinstance(dep, DependencyProg):
                # Do not know how to test this one.
                continue
            dinc_dir = dep.inc_dir
            if dinc_dir is not None:
                dinc_dir = dinc_dir.lower()
            einc_dir = self.expectations[name].inc_dir
            self.failUnlessEqual(dinc_dir, einc_dir, "%s: %s != %s" % (name, dinc_dir, einc_dir))

    def test_proper_library_path(self):
        """Ensure each dependency has found its library directory/DLL file"""
        from config_msys import DependencyProg
        
        for name, dep in dependencies.items():
            if isinstance(dep, DependencyProg):
                # Do not know how to test this one.
                continue
            dlib_dir = dep.lib_dir
            if dlib_dir is not None:
                dlib_dir = dlib_dir.lower()
            elib_dir = self.expectations[name].lib_dir
            self.failUnlessEqual(dlib_dir, elib_dir, "%s: %s != %s" % (name, dlib_dir, elib_dir))

    def test_cflags(self):
        """Ensure the cflags are properly set"""
        for name, dep in dependencies.items():
            dcflags = dep.cflags
            ecflags = self.expectations[name].cflags
            self.failUnlessEqual(dcflags, ecflags, "%s: '%s' != '%s'" % (name, dcflags, ecflags))

if __name__ == '__main__':
    unittest.main()
