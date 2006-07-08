


import unittest
from pygame import font

class FontTest( unittest.TestCase ):
    def testFontRendering( self ):
        """ 
        """

        #pygame.init()
        #screen = pygame.display.set_mode((100,100))



        import pygame
        pygame.init()
        f = pygame.font.Font(None, 20)
        s = f.render("foo", True, [0, 0, 0], [255, 255, 255])
        s = f.render("   ", True, [0, 0, 0], [255, 255, 255])
        s = f.render("", True, [0, 0, 0], [255, 255, 255])
        s = f.render("foo", False, [0, 0, 0], [255, 255, 255])
        s = f.render("   ", False, [0, 0, 0], [255, 255, 255])
        s = f.render("   ", False, [0, 0, 0])
        # null text should be 1 pixel wide.
        s = f.render("", False, [0, 0, 0], [255, 255, 255])
        self.assertEqual(s.get_size()[0], 1)


        pygame.quit()




if __name__ == '__main__':
    unittest.main()
