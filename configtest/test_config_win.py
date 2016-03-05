# program test_config_msys.py

"""Test config_msys.py for against a dummy directory structure.

This test must be performed on an MSYS console.
"""

import sys
sys.path.append('..')
import config_win

import unittest
import os

test_dir = 'testdir'
if not os.path.isdir(test_dir):
    print "Test directory %s not found." % test_dir
os.chdir(os.path.join(test_dir, 'include'))

dependencies = dict([(dep.name, dep) for dep in config_win.main()])


class RunConfigTestCase(unittest.TestCase):
    """Test dependencies returned by config_win.main()"""

    class Dependency(object):
        # Holds dependency info
        def __init__(self, lib=None, inc_dir_rel=None, lib_dir_rel=None, libs=None, cflags=None):
            if libs is None:
                if libs is not None:
                    libs = libs
                elif lib is not None:
                    libs = [lib]
                else:
                    libs = []
            if cflags is  None:
                cflags = ''
            self.libs = libs
            self.inc_dir = None
            self.lib_dir = None
            if inc_dir_rel is not None:
                self.inc_dir = '%s/%s' % ('..', inc_dir_rel)
            if lib_dir_rel is not None:
                self.lib_dir = '%s/%s' % ('..', lib_dir_rel)
            self.cflags = cflags

    # Pygame dependencies
    expectations = {
        'SDL': Dependency('SDL', 'sdl-1.2/include', 'sdl-1.2/visualc/sdl/release'),
        'FONT': Dependency('SDL_ttf', 'sdl_ttf-2.0.9', 'sdl_ttf-2.0.9/release'),
        'IMAGE': Dependency('SDL_image', 'sdl_image-1.2.6', 'sdl_image-1.2.6/visualc/release'),
        'MIXER': Dependency('SDL_mixer','sdl_mixer-1.2', 'sdl_mixer-1.2/release'),
        'PNG': Dependency('png', 'libpng-1.2.32', 'libpng-1.2.32/lib'),
        'JPEG': Dependency('jpeg', 'jpeg-6b', 'jpeg-6b/release'),
        'SCRAP': Dependency(cflags=' -luser32 -lgdi32'),
        'COPYLIB_SDL': Dependency(lib_dir_rel='sdl-1.2/visualc/sdl/release/sdl.dll'),
        'COPYLIB_SDL_ttf': Dependency(libs=['SDL', 'z'],
                                      lib_dir_rel='sdl_ttf-2.0.9/release/sdl_ttf.dll'),
        'COPYLIB_SDL_image': Dependency(libs=['SDL', 'png', 'jpeg', 'tiff'],
                                        lib_dir_rel='sdl_image-1.2.6/visualc/release/sdl_image.dll'),
        'COPYLIB_SDL_mixer': Dependency(libs=['SDL', 'vorbisfile'],
                                        lib_dir_rel='sdl_mixer-1.2/release/sdl_mixer.dll'),
        'COPYLIB_tiff': Dependency(libs=['jpeg', 'z'], lib_dir_rel='tiff-3.8.2/release/libtiff.dll'),
        'COPYLIB_png': Dependency(libs=['z'], lib_dir_rel='libpng-1.2.32/lib/libpng13.dll'),
        'COPYLIB_jpeg': Dependency(lib_dir_rel='jpeg-6b/release/jpeg.dll'),
        'COPYLIB_z': Dependency(lib_dir_rel='zlib-1.2.3/release/zlib1.dll'),
        'COPYLIB_vorbisfile': Dependency(libs=['vorbis'],
                                         lib_dir_rel='libvorbis-1.2.0/release/libvorbisfile-3.dll'),
        'COPYLIB_vorbis': Dependency(libs=['ogg'],
                                     lib_dir_rel='libvorbis-1.2.0/release/libvorbis-0.dll'),
        'COPYLIB_ogg': Dependency(lib_dir_rel='libogg-1.1.3/release/libogg-0.dll'),
        }

    def test_dependencies(self):
        """Ensure all dependencies are present"""
        self.failUnlessEqual(len(dependencies), len(self.expectations))
        for name in self.expectations:
            self.failUnless(name in dependencies, name)

    def test_found(self):
        """Ensure all dependencies were found"""
        for dep in dependencies.values():
            self.failUnless(dep.found, dep.name)

    # def test_not_found(self):
    # No easy way to test the case where something is missing

    def test_libs(self):
        """Ensure each dependency has the proper libraries"""
        for name, dep in dependencies.items():
            dlibs = set(dep.libs)
            elibs = set(self.expectations[name].libs)
            self.failUnlessEqual(dlibs, elibs, "%s: %s != %s" % (name, dlibs, elibs))

    def test_proper_include_paths(self):
        """Ensure each dependency has found its include directory"""
        for name, dep in dependencies.items():
            dinc_dir = dep.inc_dir
            if dinc_dir is not None:
                dinc_dir = dinc_dir.lower()
            einc_dir = self.expectations[name].inc_dir
            self.failUnlessEqual(dinc_dir, einc_dir, "%s: %s != %s" % (name, dinc_dir, einc_dir))

    def test_proper_library_path(self):
        """Ensure each dependency has found its library directory/DLL file"""
        for name, dep in dependencies.items():
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
            self.failUnlessEqual(dcflags, ecflags, "%s: %s != %s" % (name, dcflags, ecflags))
        
if __name__ == '__main__':
    unittest.main()
