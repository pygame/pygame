import unittest
import pygame, pygame.transform


class DisplayTest( unittest.TestCase ):
    
    def test_update( self ):
        """ see if pygame.display.update takes rects with negative values.
        """

        pygame.init()
        screen = pygame.display.set_mode((100,100))
        screen.fill((55,55,55))

        r1 = pygame.Rect(0,0,100,100)
        pygame.display.update(r1)

        r2 = pygame.Rect(-10,0,100,100)
        pygame.display.update(r2)

        r3 = pygame.Rect(-10,0,-100,-100)
        pygame.display.update(r3)





if __name__ == '__main__':
    unittest.main()
