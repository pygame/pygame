# program test_config_msys.py

"""Test config_msys.py for against a dummy directory structure.

This test must be performed on an MSYS console.
"""

import os
import os.path
import sys

# Ensure the execution environment is correct
if not ("MSYSTEM" in os.environ and os.environ["MSYSTEM"] == "MINGW32"):  # cond. and
    print "This test must be run from an MSYS console."
    sys.exit(1)

test_dir = './testdir'
if not os.path.isdir(test_dir):
    print "Test directory %s not found." % test_dir

os.environ['LOCALBASE'] = test_dir
sys.path.append('..')

import config_msys

import unittest
    
dependencies = dict([(dep.name, dep) for dep in config_msys.main()])


class RunConfigTestCase(unittest.TestCase):
    """Test dependencies returned by config_msys.main()"""

    class Dependency(object):
        # Holds dependency info
        def __init__(self, libs=None, inc_dir_rel=None, lib_dir_rel=None):
            if libs is None:
                libs = []
            self.libs = libs
            self.inc_dir = None
            self.lib_dir = None
            if inc_dir_rel is not None:
                self.inc_dir = '%s/%s' % (test_dir, inc_dir_rel)
            if lib_dir_rel is not None:
                self.lib_dir = '%s/%s' % (test_dir, lib_dir_rel)

    # Pygame dependencies
    expectations = {
        'SDL': Dependency(['SDL'], 'include/sdl', 'lib'),  # ? uses sdl-config script
        'FONT': Dependency(['SDL_ttf'], 'include/sdl', 'lib'),
        'IMAGE': Dependency(['SDL_image'], 'include/sdl', 'lib'),
        'MIXER': Dependency(['SDL_mixer'], 'include', 'lib'),  # A deviant include dir
        'SMPEG': Dependency(['smpeg'], 'include', 'lib'),  # ? uses smpeg-config script
        'PNG': Dependency(['png'], 'include', 'lib'),
        'JPEG': Dependency(['jpeg'], 'include/sdl', 'lib'),  # A deviant include dir
        'SCRAP': Dependency(['user32', 'gdi32']),
        'DLL_SDL': Dependency(lib_dir_rel='bin/sdl.dll'),
        'DLL_FONT': Dependency(lib_dir_rel='bin/sdl_ttf.dll'),  # Where DLLs likely are
        'DLL_IMAGE': Dependency(lib_dir_rel='bin/sdl_image.dll'),
        'DLL_MIXER': Dependency(lib_dir_rel='lib/sdl_mixer.dll'),  # Where the search starts
        'DLL_SMPEG': Dependency(lib_dir_rel='bin/smpeg.dll'),
        'DLL_TIFF': Dependency(lib_dir_rel='bin/libtiff.dll'),
        'DLL_PNG': Dependency(lib_dir_rel='bin/libpng13.dll'),
        'DLL_JPEG': Dependency(lib_dir_rel='bin/jpeg.dll'),
        'DLL_Z': Dependency(lib_dir_rel='bin/zlib1.dll'),
        'DLL_VORBISFILE': Dependency(lib_dir_rel='bin/libvorbisfile-3.dll'),
        'DLL_VORBIS': Dependency(lib_dir_rel='bin/libvorbis-0.dll'),
        'DLL_OGG': Dependency(lib_dir_rel='bin/libogg-0.dll'),
        }

    def test_dependencies(self):
        """Ensure all dependencies are present"""
        self.failUnlessEqual(len(dependencies), len(self.expectations))
        for name in self.expectations:
            self.failUnless(name in dependencies, name)

    def test_dll_match(self):
        """Ensure DLLs match with dll.py."""
        import dll
        
        for name in dll.regexs:
            self.failUnless('DLL_' + name in dependencies, name)

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

if __name__ == '__main__':
    unittest.main()
