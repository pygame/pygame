import unittest
import pygame, pygame.transform


class TransformTest( unittest.TestCase ):
    
    def test_scale( self ):
        """ see if set_alpha information is kept.
        """

        s = pygame.Surface((32,32))
        s.set_alpha(55)
        self.assertEqual(s.get_alpha(),55)

        s = pygame.Surface((32,32))
        s.set_alpha(55)
        s2 = pygame.transform.scale(s, (64,64))
        s3 = s.copy()
        self.assertEqual(s.get_alpha(),s3.get_alpha())
        self.assertEqual(s.get_alpha(),s2.get_alpha())



if __name__ == '__main__':
    unittest.main()
