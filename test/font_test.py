import sys
try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import pygame2
import pygame2.font as font

class FontTest (unittest.TestCase):
    
    def test_pygame2_font_find_font(self):

        # __doc__ (as of 2009-12-10) for pygame2.font.find_font:

        # find_font (name, bold=False, italic=False, ftype=None) -> str, bool, bool
        # 
        # Finds a font matching a certain family or font filename best.
        # 
        # Tries to find a font that matches the passed requirements best. The
        # *name* argument denotes a specific font or font family name. If
        # multiple fonts match that name, the *bold* and *italic* arguments
        # are used to find a font that matches the requirements best. *ftype*
        # is an optional font filetype argument to request specific font file
        # types, such as bdf or ttf fonts.

        # name, bold, italic = font.find_font ('sans')
        retval = font.find_font ("invalidfont")
        self.assertEqual (retval, None)
        
        retval = font.find_font ("sans")
        self.assertEqual (len (retval), 3)
        self.assertEqual (type (retval[0]), str)
        self.assertEqual (type (retval[1]), bool)
        self.assertEqual (type (retval[2]), bool)

    def test_pygame2_font_find_fonts(self):

        # __doc__ (as of 2009-12-10) for pygame2.font.find_fonts:

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

        retval = font.find_fonts ("invalidfont")
        self.assertEqual (retval, None)

        retval = font.find_fonts ("sans")
        self.assert_ (len (retval) != 0)
        self.assert_ (type (retval), list)

    def test_pygame2_font_get_families(self):

        # __doc__ (as of 2009-12-10) for pygame2.font.get_families:

        # get_families () -> [str, str, str, ...]
        # 
        # Gets the list of available font families.
        if sys.version in ('win32', 'darwin'):
            pass
        else:
            self.assert_ (len (font.get_families ()) != 0)
            self.assertEqual (type (font.get_families ()), list)
    
if __name__ == '__main__':
    unittest.main()
