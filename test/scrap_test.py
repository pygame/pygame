import unittest
import pygame
from pygame.locals import *
import pygame.scrap as scrap

class ScrapTest (unittest.TestCase):

    def test_scrap_mode (self):
        scrap.set_mode (SCRAP_SELECTION)
        scrap.set_mode (SCRAP_CLIPBOARD)
        self.assertRaises (ValueError, scrap.set_mode, 1099)

    def test_scrap_put_text (self):
        scrap.put (SCRAP_TEXT, "Hello world")
        self.assertEquals (scrap.get (SCRAP_TEXT), "Hello world")

        scrap.put (SCRAP_TEXT, "Another String")
        self.assertEquals (scrap.get (SCRAP_TEXT), "Another String")

    def test_scrap_put_image (self):
        sf = pygame.image.load ("examples/data/asprite.bmp")
        string = pygame.image.tostring (sf, "RGBA")
        scrap.put (SCRAP_BMP, string)
        self.assertEquals (scrap.get (SCRAP_BMP), string)
        
if __name__ == '__main__':
    pygame.init ()
    pygame.display.set_mode ((1, 1))
    scrap.init ()
    unittest.main()
