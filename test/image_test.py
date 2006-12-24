


import unittest
import pygame, pygame.image, pygame.pkgdata
import os

class ImageTest( unittest.TestCase ):
    
    def testLoadIcon( self ):
        """ see if we can load the pygame icon.
        """
        f = pygame.pkgdata.getResource("pygame_icon.bmp")
        self.assertEqual(f.mode, "rb")

        surf = pygame.image.load_basic(f)

        self.assertEqual(surf.get_at((0,0)),(5, 4, 5, 255))
        self.assertEqual(surf.get_height(),32)
        self.assertEqual(surf.get_width(),32)


    def testLoadPNG( self ):
        """ see if we can load a png.
        """
        f = os.path.join("examples", "data", "alien1.png")
        surf = pygame.image.load(f)

        f = open(os.path.join("examples", "data", "alien1.png"), "rb")
        surf = pygame.image.load(f)


    def testLoadJPG( self ):
        """ see if we can load a jpg.
        """
        f = os.path.join("examples", "data", "alien1.jpg")
        surf = pygame.image.load(f)

        f = open(os.path.join("examples", "data", "alien1.jpg"), "rb")
        surf = pygame.image.load(f)



if __name__ == '__main__':
    unittest.main()
