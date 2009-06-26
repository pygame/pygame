try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import pygame2
import pygame2.font as font

class FontTest (unittest.TestCase):
    def todo_test_pygame2_font_find_font(self):

        # __doc__ (as of 2009-06-26) for pygame2.font.find_font:

        # find_fonts(name, bold=False, italic=False, ftype=None) -> str, bool, bool
        # 
        # Finds a font matching a certain family or font filename best.
        # 
        # Tries to find a font that matches the passed requirements best. The
        # *name* argument denotes a specific font or font family name. If
        # multiple fonts match that name, the *bold* and *italic* arguments
        # are used to find a font that matches the requirements best. *ftype*
        # is an optional font filetype argument to request specific font file
        # types, such as bdf or ttf fonts.

        self.fail ()

    def todo_test_pygame2_font_find_fonts(self):

        # __doc__ (as of 2009-06-26) for pygame2.font.find_fonts:

        # find_fonts(name, bold=False, italic=False, ftype=None) -> [ (str, bool, bool), ... ]
        # 
        # Finds all fonts matching a certain family or font filename.
        # 
        # Tries to find all fonts that match the passed requirements best. The
        # *name* argument denotes a specific font or font family name. If
        # multiple fonts match that name, the *bold* and *italic* arguments
        # are used to find the fonts that match the requirements best. *ftype*
        # is an optional font filetype argument to request specific font file
        # types, such as bdf or ttf fonts.
        # 
        # All found fonts are returned as list

        self.fail() 

    def todo_test_pygame2_font_get_families(self):

        # __doc__ (as of 2009-06-26) for pygame2.font.get_families:

        # get_families () -> [str, str, str, ...]
        # 
        # Gets the list of available font families.

        self.fail() 
