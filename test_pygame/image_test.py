


import unittest
import pygame, pygame.image, pygame.pkgdata

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


if __name__ == '__main__':
    unittest.main()
