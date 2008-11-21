# program test_dll.py

"""A unit test on the dll.py module that confirms file matching patterns"""

import sys

sys.path.append('..')

import dll

import unittest

class MatchTestCase(unittest.TestCase):

    test_cases = [
        ('SDL', ['SDL.dll', 'sdl.DLL', 'libsdl.dll'], ['sdl.dll.a']),
        ('MIXER', ['SDL_mixer.dll', 'sdl_MIXER.DLL', 'libsdl_mixer.dll'], ['sdl_mixer.dll.a']),
        ('IMAGE', ['SDL_image.dll', 'sdl_IMAGE.DLL', 'libsdl_image.dll'], ['sdl_image.dll.a']),
        ('FONT', ['SDL_ttf.dll', 'sdl_TTF.DLL', 'libsdl_ttf.dll'], ['sdl_ttf.dll.a']),
        ('SMPEG', ['smpeg.dll', 'SMPEG.DLL', 'libsmpeg.dll'], ['smpeg.dll.a']),
        ('TIFF', ['tiff.dll', 'TIFF.DLL', 'libtiff.dll'], ['tiff.dll.a']),
        ('JPEG', ['jpeg.dll', 'JPEG.DLL', 'libjpeg.dll'], ['jpeg.dll.a']),
        ('PNG', ['libpng13.dll', 'LIBPNG13.DLL', 'libpng12.dll', 'png12.dll', 'png13.dll', 'libpng12-0.dll'],
                ['libpng.dll', 'libpng13.dll.a', 'libpng12.dll.a']),
        ('Z', ['zlib1.dll', 'ZLIB1.DLL'], ['z.dll', 'libzlib1.dll', 'zlib1.dll.a']),
        ('VORBIS', ['vorbis.dll', 'VORBIS.DLL', 'libvorbis-0.dll'], ['libvorbis-1.dll', 'libvorbis-0.dll.a']),
        ('VORBISFILE', ['vorbisfile.dll', 'VORBISFILE.DLL', 'libvorbisfile-3.dll'],
                       ['libvorbisfile-0.dll', 'libvorbisfile-3.dll.a']),
        ('OGG', ['ogg.dll', 'OGG.DLL', 'libogg-0.dll'], ['libogg-1.dll', 'libogg-0.dll.a']),
        ('FREETYPE', ['freetype.dll', 'FREETYPE.DLL', 'libfreetype-6.dll'],
         ['libfreetype.dll.a']),
        ]

    def test_compat(self):
        """Validate the test cases"""
        self.failUnlessEqual(len(self.test_cases), len(dll.regexs))
        for name, valid_files, invalid_files in self.test_cases:
            self.failUnless(name in  dll.regexs, name)

    def test_match(self):
        """Ensure certain file names match"""
        for name, valid_files, invalid_files in self.test_cases:
            test = dll.tester(name)
            for f in valid_files:
                self.failUnless(test(f), f)

    def test_failed_match(self):
        """Ensure certain file names do not match"""
        for name, valid_files, invalid_files in self.test_cases:
            test = dll.tester(name)
            for f in invalid_files:
                self.failUnless(not test(f), f)

class DependencyLookupTestCase(unittest.TestCase):
    def test_no_dependencies(self):
        """Ensure no dependencies are returned for a library with non"""
        self.failUnlessEqual(list(dll.dependencies(['SDL'])), ['SDL'])

    def test_not_found(self):
        """Ensure an empty dependency list is returned for an unrecognized library"""
        self.failUnless(not dll.dependencies(['?']))

    def test_multiple_dependencies(self):
        """Ensure dependencies are recursively traced"""
        expected_libs = ['VORBISFILE', 'VORBIS', 'OGG']
        libs = dll.dependencies(['VORBISFILE'])
        self.failUnlessEqual(len(libs), len(expected_libs))
        for lib in expected_libs:
            self.failUnless(lib in libs)

    def test_multiple_libs(self):
        """Ensure mutliple libraries in a list are handled"""
        expected_libs = ['SDL', 'Z']  # Chosen for not having dependencies
        libs = dll.dependencies(expected_libs)
        self.failUnlessEqual(len(libs), len(expected_libs))
        for lib in expected_libs:
            self.failUnless(lib in libs)

    def test_no_libs(self):
        """Check special case of an empty library list"""
        self.failUnless(not dll.dependencies([]))

class RootNameLookupTestCase(unittest.TestCase):
    def test_found(self):
        """Ensure name -> file root name works for at least one case"""
        self.failUnlessEqual(dll.name_to_root('FONT'), 'SDL_ttf')

    def test_not_found(self):
        """Ensure an exception is raised for an unrecognized name"""
        def test():
            dll.name_to_root('*')
        self.failUnlessRaises(KeyError, test)
        
if __name__ == '__main__':
    unittest.main()
