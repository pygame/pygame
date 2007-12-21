import unittest
import pygame

class PixelArrayTest (unittest.TestCase):

    def test_pixel_array (self):
        sf = pygame.Surface ((10, 20), 0, 32)
        sf.fill ((0, 0, 0))
        ar = pygame.PixelArray (sf)

        if sf.mustlock():
            self.assertTrue (sf.get_locked())
        self.assertEqual (len (ar), 10)
        del ar
        if sf.mustlock():
            self.assertFalse (sf.get_locked())

    # Sequence interfaces
    def test_get_column (self):
        sf = pygame.Surface ((10, 20), 0, 32)
        sf.fill ((0, 0, 0))
        ar = pygame.PixelArray (sf)

        ar2 = ar[0]
        self.assertEqual (len(ar2), 20)

        ar2 = ar[-1]
        self.assertEqual (len(ar2), 20)

    def test_get_pixel (self):
        sf = pygame.Surface ((10, 20), 0, 32)
        sf.fill ((0, 0, 255))
        for x in xrange(20):
            sf.set_at((1,x), 0x000011)
        for x in xrange(10):
            sf.set_at((x,1), 0x000011)

        ar = pygame.PixelArray (sf)
        
        ar2 = ar[0][0]
        self.assertEqual (ar2, 0x0000FF)

        ar2 = ar[1][0]
        self.assertEqual (ar2, 0x000011)

        ar2 = ar[-4][1]
        self.assertEqual (ar2, 0x000011)
        ar2 = ar[-4][5]
        self.assertEqual (ar2, 0x0000FF)

    def test_set_pixel (self):
        sf = pygame.Surface ((10, 20), 0, 32)
        sf.fill ((0, 0, 0))
        ar = pygame.PixelArray (sf)

        ar[0][0] = 0x00FF00
        self.assertEqual (ar[0][0], 0x00FF00)

        ar[1][1] = (128, 128, 128)
        self.assertEqual (ar[1][1], 0x808080)

        ar[-1][-1] = (128, 128, 128)
        self.assertEqual (ar[9][19], 0x808080)

        ar[-2][-2] = (128, 128, 128)
        self.assertEqual (ar[8][-2], 0x808080)

    def test_set_column (self):
        sf = pygame.Surface ((6, 8), 0, 32)
        sf.fill ((0, 0, 0))
        ar = pygame.PixelArray (sf)

        sf2 = pygame.Surface ((6, 8), 0, 32)
        sf2.fill ((0, 255, 255))
        ar2 = pygame.PixelArray (sf2)

        # Test single value assignment
        ar[2] = 0x808080
        self.assertEqual (ar[2][0], 0x808080)
        self.assertEqual (ar[2][1], 0x808080)

        ar[-1] = 0x00FFFF
        self.assertEqual (ar[5][0], 0x00FFFF)
        self.assertEqual (ar[-1][1], 0x00FFFF)

        ar[-2] = (255, 255, 0)
        self.assertEqual (ar[4][0], 0xFFFF00)
        self.assertEqual (ar[-2][1], 0xFFFF00)

        # Test list assignment.
        ar[0] = [0xFFFFFF] * 8
        self.assertEqual (ar[0][0], 0xFFFFFF)
        self.assertEqual (ar[0][1], 0xFFFFFF)

        # Test tuple assignment.
        ar[1] = (0xCC00CC, 0x111111, 0xCC00CC, 0x111111,
                 0xCC00CC, 0x111111, 0xCC00CC, 0x111111)
        self.assertEqual (ar[1][0], 0xCC00CC)
        self.assertEqual (ar[1][1], 0x111111)

        # Test pixel array assignment.
        ar[1] = ar2[3]
        self.assertEqual (ar[1][0], 0x00FFFF)
        self.assertEqual (ar[1][1], 0x00FFFF)

    def test_get_slice (self):
        sf = pygame.Surface ((10, 20), 0, 32)
        sf.fill ((0, 0, 0))
        ar = pygame.PixelArray (sf)

        self.assertEqual (len (ar[0:2]), 2)
        self.assertEqual (len (ar[3:7][3]), 20)

        self.assertRaises (IndexError, ar.__getslice__, 0, 0)
        self.assertRaises (IndexError, ar.__getslice__, 9, 9)

        # Has to resolve to ar[7:8]
        self.assertEqual (len (ar[-3:-2]), 20)

##     def test_set_slice (self):
##         sf = pygame.Surface ((6, 8), 0, 32)
##         sf.fill ((0, 0, 0))
##         ar = pygame.PixelArray (sf)

##         # Test single value assignment
##         ar[0:2] = 0x808080
##         print ar
##         self.assertEqual (ar[0][0], 0x808080)
##         self.assertEqual (ar[0][1], 0x808080)
##         self.assertEqual (ar[1][0], 0x808080)
##         self.assertEqual (ar[1][1], 0x808080)

##         ar[-2:-1] = 0x00FFFF
##         print ar
##         self.assertEqual (ar[4][0], 0x00FFFF)
##         self.assertEqual (ar[-1][1], 0x00FFFF)

##         ar[-2:-1] = (255, 255, 0)
##         print ar
##         self.assertEqual (ar[4][0], 0xFFFF00)
##         self.assertEqual (ar[-1][1], 0xFFFF00)

##         # Test list assignment.
##         ar[2:4] = [0xFFFFFF] * 16
##         print ar
##         self.assertEqual (ar[2][0], 0xFFFFFF)
##         self.assertEqual (ar[3][1], 0xFFFFFF)

##         # Test tuple assignment.
##         ar[1:4] = (0xCC00CC, 0x111111, 0xCC00CC, 0x111111,
##                    0xCC00CC, 0x111111, 0xCC00CC, 0x111111) * 3
##         print ar
##         self.assertEqual (ar[1][0], 0xCC00CC)
##         self.assertEqual (ar[1][1], 0x111111)

##         # Test pixel array assignment.
##         ar[1] = ar[3]
##         print ar
##         self.assertEqual (ar[1][0], 0x000000)
##         self.assertEqual (ar[1][1], 0x000000)

if __name__ == '__main__':
    unittest.main()
