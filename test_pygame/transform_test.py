import unittest
import pygame, pygame.transform


class TransformTest( unittest.TestCase ):
    
    def test_scale_alpha( self ):
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


    def test_scale_destination( self ):
        """ see if the destination surface can be passed in to use.
        """

        s = pygame.Surface((32,32))
        s2 = pygame.transform.scale(s, (64,64))
        s3 = s2.copy()

        s3 = pygame.transform.scale(s, (64,64), s3)
        pygame.transform.scale(s, (64,64), s2)

        # the wrong size surface is past in.  Should raise an error.
        self.assertRaises(ValueError, pygame.transform.scale, s, (33,64), s3)




if __name__ == '__main__':
    unittest.main()
