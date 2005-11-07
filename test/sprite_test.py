


import unittest
from pygame import sprite

class SpriteTest( unittest.TestCase ):
    def testAbstractGroup_has( self ):
        """ See if abstractGroup has works as expected.
        """
        ag = sprite.AbstractGroup()
        ag2 = sprite.AbstractGroup()
        s1 = sprite.Sprite(ag)
        s2 = sprite.Sprite(ag)
        s3 = sprite.Sprite(ag2)
        s4 = sprite.Sprite(ag2)

        self.assertEqual(True, s1 in ag )

        self.assertEqual(True, ag.has(s1) )

        self.assertEqual(True, ag.has([s1, s2]) )

        # see if one of them not being in there.
        self.assertNotEqual(True, ag.has([s1, s2, s3]) )

        # see if a second AbstractGroup works.
        self.assertEqual(True, ag2.has(s3) )




if __name__ == '__main__':
    unittest.main()
