# program test_config_msys.py

"""Test config_msys.py for against a dummy directory structure.

This test must be performed on an MSYS console.
"""

import sys
sys.path.append('..')
import config_win
import dll

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
        def __init__(self, name=None, inc_dir_rel=None, lib_dir_rel=None, libs=None):
            if libs is None:
                if name is None:
                    libs = []
                else:
                    libs = [dll.name_to_root(name)]
            self.libs = libs
            self.inc_dir = None
            self.lib_dir = None
            if inc_dir_rel is not None:
                self.inc_dir = '%s/%s' % ('..', inc_dir_rel)
            if lib_dir_rel is not None:
                self.lib_dir = '%s/%s' % ('..', lib_dir_rel)

    # Pygame dependencies
    expectations = {
        'SDL': Dependency('SDL', 'sdl-1.2.12/include', 'sdl-1.2.12/visualc/sdl/release'),
        'FONT': Dependency('FONT', 'sdl_ttf-2.0.9', 'sdl_ttf-2.0.9/release'),
        'IMAGE': Dependency('IMAGE', 'sdl_image-1.2.6', 'sdl_image-1.2.6/visualc/release'),
        'MIXER': Dependency('MIXER', 'sdl_mixer-1.2.8', 'sdl_mixer-1.2.8/release'),
        'SMPEG': Dependency('SMPEG', 'smpeg', 'smpeg/release'),
        'PNG': Dependency('PNG', 'libpng-1.2.19', 'libpng-1.2.19/lib'),
        'JPEG': Dependency('JPEG', 'jpeg-6b', 'jpeg-6b/release'),
        'SCRAP': Dependency(libs=['user32', 'gdi32']),
        'COPYLIB_SDL': Dependency('SDL',
                                  lib_dir_rel='sdl-1.2.12/visualc/sdl/release/sdl.dll'),
        'COPYLIB_FONT': Dependency('FONT',
                                   lib_dir_rel='sdl_ttf-2.0.9/release/sdl_ttf.dll'),
        'COPYLIB_IMAGE': Dependency('IMAGE',
                                    lib_dir_rel='sdl_image-1.2.6/visualc/release/sdl_image.dll'),
        'COPYLIB_MIXER': Dependency('MIXER',
                                    lib_dir_rel='sdl_mixer-1.2.8/release/sdl_mixer.dll'),
        'COPYLIB_SMPEG': Dependency('SMPEG', lib_dir_rel='smpeg/release/smpeg.dll'),
        'COPYLIB_TIFF': Dependency('TIFF', lib_dir_rel='tiff-3.8.2/release/libtiff.dll'),
        'COPYLIB_PNG': Dependency('PNG', lib_dir_rel='libpng-1.2.19/lib/libpng13.dll'),
        'COPYLIB_JPEG': Dependency('JPEG', lib_dir_rel='jpeg-6b/release/jpeg.dll'),
        'COPYLIB_Z': Dependency('Z', lib_dir_rel='zlib-1.2.3/release/zlib1.dll'),
        'COPYLIB_VORBISFILE': Dependency('VORBISFILE',
                                         lib_dir_rel='libvorbis-1.2.0/release/libvorbisfile-3.dll'),
        'COPYLIB_VORBIS': Dependency('VORBIS',
                                     lib_dir_rel='libvorbis-1.2.0/release/libvorbis-0.dll'),
        'COPYLIB_OGG': Dependency('OGG', lib_dir_rel='libogg-1.1.3/release/libogg-0.dll'),
        }

    def test_dependencies(self):
        """Ensure all dependencies are present"""
        self.failUnlessEqual(len(dependencies), len(self.expectations))
        for name in self.expectations:
            self.failUnless(name in dependencies, name)

    def test_dll_match(self):
        """Ensure DLLs match with dll.py."""
        for name in dll.regexs:
            self.failUnless('COPYLIB_' + name in dependencies, name)

    def test_found(self):
        """Ensure all dependencies were found"""
        for dep in dependencies.values():
            self.failUnless(dep.found, dep.name)

    # def test_not_found(self):
    # No easy way to test the case where something is missing

    def test_libs(self):
        """Ensure each dependency has the proper libraries"""
        for name, dep in dependencies.items():
            dlibs = dep.libs
            elibs = self.expectations[name].libs
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

if __name__ == '__main__':
    unittest.main()
