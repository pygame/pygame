import unittest
import pygame2
import pygame2.freetype.base as base

class FreeTypeTest(unittest.TestCase):

    def test_pygame2_freetype_base_get_version(self):

        # __doc__ (as of 2009-05-17) for pygame2.freetype.base.get_version:

        # get_version () -> tuple
        # 
        # Gets the version of the FreeType2 library which was used to build
        # the 'freetype' module.

        version = base.get_version()
        self.assertEqual(len(version), 3)
        self.assertEqual(version[0], 2)

    def test_pygame2_freetype_base_init(self):

        # __doc__ (as of 2009-05-17) for pygame2.freetype.base.init:

        # init () -> None
        # 
        # Initializes the underlying FreeType 2 library.  This method
        # must be called before trying to use any of the functionality
        # of the 'freetype' module.

        # init()
        base.init()
        self.assertEqual(base.was_init(), True)
        
        # quit
        base.quit()
        self.assertEqual(base.was_init(), False)

if __name__ == '__main__':
    unittest.main()
