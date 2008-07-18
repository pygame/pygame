import test_utils
import test.unittest as unittest

from test_utils import test_not_implemented

import pygame
import pygame.scrap as scrap

class ScrapModuleTest(unittest.TestCase):    
    def test_contains(self):

        # __doc__ (as of 2008-06-25) for pygame.scrap.contains:

          # scrap.contains (type) -> bool
          # Checks, whether a certain type is available in the clipboard.

        self.assert_(test_not_implemented()) 

    def test_get(self):

        # __doc__ (as of 2008-06-25) for pygame.scrap.get:

          # scrap.get (type) -> string
          # Gets the data for the specified type from the clipboard.

        self.assert_(test_not_implemented()) 

    def test_get_types(self):

        # __doc__ (as of 2008-06-25) for pygame.scrap.get_types:

          # scrap.get_types () -> list
          # Gets a list of the available clipboard types.

        self.assert_(test_not_implemented()) 

    def test_init(self):

        # __doc__ (as of 2008-06-25) for pygame.scrap.init:

          # scrap.init () -> None
          # Initializes the scrap module.

        self.assert_(test_not_implemented()) 

    def test_lost(self):

        # __doc__ (as of 2008-06-25) for pygame.scrap.lost:

          # scrap.lost() -> bool
          # Checks whether the clipboard is currently owned by the application.

        self.assert_(test_not_implemented()) 

    def test_set_mode (self):
        scrap.set_mode (pygame.SCRAP_SELECTION)
        scrap.set_mode (pygame.SCRAP_CLIPBOARD)
        self.assertRaises (ValueError, scrap.set_mode, 1099)

    def test_scrap_put_text (self):
        scrap.put (pygame.SCRAP_TEXT, "Hello world")
        self.assertEquals (scrap.get (pygame.SCRAP_TEXT), "Hello world")

        scrap.put (pygame.SCRAP_TEXT, "Another String")
        self.assertEquals (scrap.get (pygame.SCRAP_TEXT), "Another String")

    def test_scrap_put_image (self):
        sf = pygame.image.load ("examples/data/asprite.bmp")
        string = pygame.image.tostring (sf, "RGBA")
        scrap.put (pygame.SCRAP_BMP, string)
        self.assertEquals (scrap.get(pygame.SCRAP_BMP), string)

    def test_put (self):
        scrap.put ("arbitrary buffer", "buf")
        r = scrap.get ("arbitrary buffer")
        self.assertEquals (r, "buf")

if __name__ == '__main__':
    pygame.init ()
    pygame.display.set_mode ((1, 1))
    scrap.init ()
    unittest.main()