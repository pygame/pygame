import unittest
import pygame

class PixelArrayTest (unittest.TestCase):

    def test_pixel_array (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 20), 0, bpp)
            sf.fill ((0, 0, 0))
            ar = pygame.PixelArray (sf)

            if sf.mustlock():
                self.assertTrue (sf.get_locked ())

            self.assertEqual (len (ar), 10)
            del ar

            if sf.mustlock():
                self.assertFalse (sf.get_locked ())

    # Sequence interfaces
    def test_get_column (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 20), 0, bpp)
            sf.fill ((0, 0, 0))
            ar = pygame.PixelArray (sf)

            ar2 = ar[0]
            self.assertEqual (len(ar2), 20)
            
            ar2 = ar[-1]
            self.assertEqual (len(ar2), 20)

    def test_get_pixel (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 20), 0, bpp)
            sf.fill ((0, 0, 255))
            for x in xrange (20):
                sf.set_at ((1, x), (0, 0, 11))
            for x in xrange (10):
                sf.set_at ((x, 1), (0, 0, 11))

            ar = pygame.PixelArray (sf)

            ar2 = ar[0][0]
            self.assertEqual (ar2, sf.map_rgb ((0, 0, 255)))
        
            ar2 = ar[1][0]
            self.assertEqual (ar2, sf.map_rgb ((0, 0, 11)))
            
            ar2 = ar[-4][1]
            self.assertEqual (ar2, sf.map_rgb ((0, 0, 11)))
        
            ar2 = ar[-4][5]
            self.assertEqual (ar2, sf.map_rgb ((0, 0, 255)))

    def test_set_pixel (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 20), 0, bpp)
            sf.fill ((0, 0, 0))
            ar = pygame.PixelArray (sf)

            ar[0][0] = (0, 255, 0)
            self.assertEqual (ar[0][0], sf.map_rgb ((0, 255, 0)))

            ar[1][1] = (128, 128, 128)
            self.assertEqual (ar[1][1], sf.map_rgb ((128, 128, 128)))
            
            ar[-1][-1] = (128, 128, 128)
            self.assertEqual (ar[9][19], sf.map_rgb ((128, 128, 128)))
            
            ar[-2][-2] = (128, 128, 128)
            self.assertEqual (ar[8][-2], sf.map_rgb ((128, 128, 128)))

    def test_set_column (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((6, 8), 0, bpp)
            sf.fill ((0, 0, 0))
            ar = pygame.PixelArray (sf)

            sf2 = pygame.Surface ((6, 8), 0, bpp)
            sf2.fill ((0, 255, 255))
            ar2 = pygame.PixelArray (sf2)

            # Test single value assignment
            ar[2] = (128, 128, 128)
            self.assertEqual (ar[2][0], sf.map_rgb ((128, 128, 128)))
            self.assertEqual (ar[2][1], sf.map_rgb ((128, 128, 128)))
        
            ar[-1] = (0, 255, 255)
            self.assertEqual (ar[5][0], sf.map_rgb ((0, 255, 255)))
            self.assertEqual (ar[-1][1], sf.map_rgb ((0, 255, 255)))
        
            ar[-2] = (255, 255, 0)
            self.assertEqual (ar[4][0], sf.map_rgb ((255, 255, 0)))
            self.assertEqual (ar[-2][1], sf.map_rgb ((255, 255, 0)))
        
            # Test list assignment.
            ar[0] = [(255, 255, 255)] * 8
            self.assertEqual (ar[0][0], sf.map_rgb ((255, 255, 255)))
            self.assertEqual (ar[0][1], sf.map_rgb ((255, 255, 255)))
            
            # Test tuple assignment.
            ar[1] = ((204, 0, 204), (17, 17, 17), (204, 0, 204), (17, 17, 17),
                     (204, 0, 204), (17, 17, 17), (204, 0, 204), (17, 17, 17))
            self.assertEqual (ar[1][0], sf.map_rgb ((204, 0, 204)))
            self.assertEqual (ar[1][1], sf.map_rgb ((17, 17, 17)))
        
            # Test pixel array assignment.
            ar[1] = ar2[3]
            self.assertEqual (ar[1][0], sf.map_rgb ((0, 255, 255)))
            self.assertEqual (ar[1][1], sf.map_rgb ((0, 255, 255)))

    def test_get_slice (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 20), 0, bpp)
            sf.fill ((0, 0, 0))
            ar = pygame.PixelArray (sf)
        
            self.assertEqual (len (ar[0:2]), 2)
            self.assertEqual (len (ar[3:7][3]), 20)
        
            self.assertRaises (IndexError, ar.__getslice__, 0, 0)
            self.assertRaises (IndexError, ar.__getslice__, 9, 9)
        
            # Has to resolve to ar[7:8]
            self.assertEqual (len (ar[-3:-2]), 20)

            # Try assignments.

            # 2D assignment.
            ar[2:5] = (255, 255, 255)
            self.assertEqual (ar[3][3], sf.map_rgb ((255, 255, 255)))

            # 1D assignment
            ar[3][3:7] = (10, 10, 10)
            self.assertEqual (ar[3][5], sf.map_rgb ((10, 10, 10)))

    def test_contains (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 20), 0, bpp)
            sf.fill ((0, 0, 0))
            sf.set_at ((8, 8), (255, 255, 255))

            ar = pygame.PixelArray (sf)
            self.assertTrue ((0, 0, 0) in ar)
            self.assertTrue ((255, 255, 255) in ar)
            self.assertFalse ((255, 255, 0) in ar)
            self.assertFalse (0x0000ff in ar)

            # Test sliced array
            self.assertTrue ((0, 0, 0) in ar[8])
            self.assertTrue ((255, 255, 255) in ar[8])
            self.assertFalse ((255, 255, 0) in ar[8])
            self.assertFalse (0x0000ff in ar[8])

    def test_get_surface (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 20), 0, bpp)
            sf.fill ((0, 0, 0))
            ar = pygame.PixelArray (sf)
            self.assertEqual (sf, ar.surface)

    def test_set_slice (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((6, 8), 0, 32)
            sf.fill ((0, 0, 0))
            ar = pygame.PixelArray (sf)

            # Test single value assignment
            val = sf.map_rgb ((128, 128, 128))
            ar[0:2] = val
            self.assertEqual (ar[0][0], val)
            self.assertEqual (ar[0][1], val)
            self.assertEqual (ar[1][0], val)
            self.assertEqual (ar[1][1], val)

            val = sf.map_rgb ((0, 255, 255))
            ar[-3:-1] = val
            self.assertEqual (ar[3][0], val)
            self.assertEqual (ar[-2][1], val)

            val = sf.map_rgb ((255, 255, 0))
            ar[-3:] = (255, 255, 0)
            
            self.assertEqual (ar[4][0], val)
            self.assertEqual (ar[-1][1], val)

            # Test list assignment.
            val = sf.map_rgb ((0, 255, 0))
            ar[2:4] = [val] * 8
            self.assertEqual (ar[2][0], val)
            self.assertEqual (ar[3][1], val)

if __name__ == '__main__':
    unittest.main()
