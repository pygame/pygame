import unittest
import pygame

class FontTest( unittest.TestCase ):
    def testFontRendering( self ):
        """ 
        """
        #print __file__
        pygame.font.init ()
        f = pygame.font.Font(None, 20)
        s = f.render("foo", True, [0, 0, 0], [255, 255, 255])
        s = f.render("xxx", True, [0, 0, 0], [255, 255, 255])
        s = f.render("", True, [0, 0, 0], [255, 255, 255])
        s = f.render("foo", False, [0, 0, 0], [255, 255, 255])
        s = f.render("xxx", False, [0, 0, 0], [255, 255, 255])
        s = f.render("xxx", False, [0, 0, 0])
        s = f.render("   ", False, [0, 0, 0])
        s = f.render("   ", False, [0, 0, 0], [255, 255, 255])
        # null text should be 1 pixel wide.
        s = f.render("", False, [0, 0, 0], [255, 255, 255])
        self.assertEqual(s.get_size()[0], 1)
        #print "fonttest done"
        #pygame.font.quit ()

if __name__ == '__main__':
    unittest.main()
