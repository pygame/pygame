import unittest
import pygame
import pygame.scrap as scrap

class ScrapTest (unittest.TestCase):

    def test_scrap_mode (self):
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
        self.assertEquals (scrap.get (pygame.SCRAP_BMP), string)

    def test_scrap_put (self):
        scrap.put ("arbitrary buffer", "buf")
        self.assertEquals (scrap.get ("arbitrary buffer"), "buf")

if __name__ == '__main__':
    pygame.init ()
    pygame.display.set_mode ((1, 1))
    scrap.init ()
    unittest.main()
