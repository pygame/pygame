#################################### IMPORTS ###################################

import test_utils, unittest
from test_utils import test_not_implemented

################################################################################

class Test(unittest.TestCase):
    pass

    def test_SysFont(self):

        # __doc__ (as of 2008-06-25) for pygame.sysfont.SysFont:

          # pygame.font.SysFont(name, size, bold=False, italic=False) -> Font
          # create a pygame Font from system font resources
          # 
          # This will search the system fonts for the given font
          # name. You can also enable bold or italic styles, and
          # the appropriate system font will be selected if available.
          # 
          # This will always return a valid Font object, and will
          # fallback on the builtin pygame font if the given font
          # is not found.
          # 
          # Name can also be a comma separated list of names, in
          # which case set of names will be searched in order. Pygame
          # uses a small set of common font aliases, if the specific
          # font you ask for is not available, a reasonable alternative
          # may be used.

        self.assert_(test_not_implemented()) 

    def test_create_aliases(self):

        # __doc__ (as of 2008-06-25) for pygame.sysfont.create_aliases:

          # 

        self.assert_(test_not_implemented()) 

    def test_get_fonts(self):

        # __doc__ (as of 2008-06-25) for pygame.sysfont.get_fonts:

          # pygame.font.get_fonts() -> list
          # get a list of system font names
          # 
          # Returns the list of all found system fonts. Note that
          # the names of the fonts will be all lowercase with spaces
          # removed. This is how pygame internally stores the font
          # names for matching.

        self.assert_(test_not_implemented()) 

    def test_initsysfonts(self):

        # __doc__ (as of 2008-06-25) for pygame.sysfont.initsysfonts:

          # 

        self.assert_(test_not_implemented()) 

    def test_initsysfonts_darwin(self):

        # __doc__ (as of 2008-06-25) for pygame.sysfont.initsysfonts_darwin:

          # 

        self.assert_(test_not_implemented()) 

    def test_initsysfonts_unix(self):

        # __doc__ (as of 2008-06-25) for pygame.sysfont.initsysfonts_unix:

          # 

        self.assert_(test_not_implemented()) 

    def test_initsysfonts_win32(self):

        # __doc__ (as of 2008-06-25) for pygame.sysfont.initsysfonts_win32:

          # 

        self.assert_(test_not_implemented()) 

    def test_match_font(self):

        # __doc__ (as of 2008-06-25) for pygame.sysfont.match_font:

          # pygame.font.match_font(name, bold=0, italic=0) -> name
          # find the filename for the named system font
          # 
          # This performs the same font search as the SysFont()
          # function, only it returns the path to the TTF file
          # that would be loaded. The font name can be a comma
          # separated list of font names to try.
          # 
          # If no match is found, None is returned.

        self.assert_(test_not_implemented()) 


################################################################################

if __name__ == '__main__':
    test_utils.get_fail_incomplete_tests_option()
    unittest.main()
