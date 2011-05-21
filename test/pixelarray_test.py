import sys
if __name__ == '__main__':
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

if is_pygame_pkg:
    from pygame.tests.test_utils import test_not_implemented, unittest
else:
    from test.test_utils import test_not_implemented, unittest
import pygame
from pygame.compat import xrange_

PY3 = sys.version_info >= (3, 0, 0)

class PixelArrayTypeTest (unittest.TestCase):
    def todo_test_compare(self):
        # __doc__ (as of 2008-06-25) for pygame.pixelarray.PixelArray.compare:

          # PixelArray.compare (array, distance=0, weights=(0.299, 0.587, 0.114)): Return PixelArray
          # Compares the PixelArray with another one.

        self.fail()

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
            sf = pygame.Surface ((6, 8), 0, bpp)
            sf.fill ((0, 0, 255))
            val = sf.map_rgb ((0, 0, 255))
            ar = pygame.PixelArray (sf)

            ar2 = ar.__getitem__ (1)
            self.assertEqual (len(ar2), 8)
            self.assertEqual (ar2.__getitem__ (0), val)
            self.assertEqual (ar2.__getitem__ (1), val)
            self.assertEqual (ar2.__getitem__ (2), val)

            ar2 = ar.__getitem__ (-1)
            self.assertEqual (len(ar2), 8)
            self.assertEqual (ar2.__getitem__ (0), val)
            self.assertEqual (ar2.__getitem__ (1), val)
            self.assertEqual (ar2.__getitem__ (2), val)

    def test_get_pixel (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 20), 0, bpp)
            sf.fill ((0, 0, 255))
            for x in xrange_ (20):
                sf.set_at ((1, x), (0, 0, 11))
            for x in xrange_ (10):
                sf.set_at ((x, 1), (0, 0, 11))

            ar = pygame.PixelArray (sf)

            ar2 = ar.__getitem__ (0).__getitem__ (0)
            self.assertEqual (ar2, sf.map_rgb ((0, 0, 255)))
        
            ar2 = ar.__getitem__ (1).__getitem__ (0)
            self.assertEqual (ar2, sf.map_rgb ((0, 0, 11)))
            
            ar2 = ar.__getitem__ (-4).__getitem__ (1)
            self.assertEqual (ar2, sf.map_rgb ((0, 0, 11)))
        
            ar2 = ar.__getitem__ (-4).__getitem__ (5)
            self.assertEqual (ar2, sf.map_rgb ((0, 0, 255)))

    def test_set_pixel (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 20), 0, bpp)
            sf.fill ((0, 0, 0))
            ar = pygame.PixelArray (sf)

            ar.__getitem__ (0).__setitem__ (0, (0, 255, 0))
            self.assertEqual (ar[0][0], sf.map_rgb ((0, 255, 0)))

            ar.__getitem__ (1).__setitem__ (1, (128, 128, 128))
            self.assertEqual (ar[1][1], sf.map_rgb ((128, 128, 128)))
            
            ar.__getitem__(-1).__setitem__ (-1, (128, 128, 128))
            self.assertEqual (ar[9][19], sf.map_rgb ((128, 128, 128)))
            
            ar.__getitem__ (-2).__setitem__ (-2, (128, 128, 128))
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
            ar.__setitem__ (2, (128, 128, 128))
            self.assertEqual (ar[2][0], sf.map_rgb ((128, 128, 128)))
            self.assertEqual (ar[2][1], sf.map_rgb ((128, 128, 128)))
        
            ar.__setitem__ (-1, (0, 255, 255))
            self.assertEqual (ar[5][0], sf.map_rgb ((0, 255, 255)))
            self.assertEqual (ar[-1][1], sf.map_rgb ((0, 255, 255)))
        
            ar.__setitem__ (-2, (255, 255, 0))
            self.assertEqual (ar[4][0], sf.map_rgb ((255, 255, 0)))
            self.assertEqual (ar[-2][1], sf.map_rgb ((255, 255, 0)))
        
            # Test list assignment.
            ar.__setitem__ (0, [(255, 255, 255)] * 8)
            self.assertEqual (ar[0][0], sf.map_rgb ((255, 255, 255)))
            self.assertEqual (ar[0][1], sf.map_rgb ((255, 255, 255)))
            
            # Test tuple assignment.
            ar.__setitem__ (1, ((204, 0, 204), (17, 17, 17), (204, 0, 204),
                                (17, 17, 17), (204, 0, 204), (17, 17, 17),
                                (204, 0, 204), (17, 17, 17)))
            self.assertEqual (ar[1][0], sf.map_rgb ((204, 0, 204)))
            self.assertEqual (ar[1][1], sf.map_rgb ((17, 17, 17)))
            self.assertEqual (ar[1][2], sf.map_rgb ((204, 0, 204)))
        
            # Test pixel array assignment.
            ar.__setitem__ (1, ar2.__getitem__ (3))
            self.assertEqual (ar[1][0], sf.map_rgb ((0, 255, 255)))
            self.assertEqual (ar[1][1], sf.map_rgb ((0, 255, 255)))

    def test_get_slice (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 20), 0, bpp)
            sf.fill ((0, 0, 0))
            ar = pygame.PixelArray (sf)

            if PY3:
                self.assertEqual (len (ar[0:2]), 2)
                self.assertEqual (len (ar[3:7][3]), 20)

                self.assertEqual (ar[0:0], None)
                self.assertEqual (ar[5:5], None)
                self.assertEqual (ar[9:9], None)
            else:
                self.assertEqual (len (ar.__getslice__ (0, 2)), 2)
                self.assertEqual (len (ar.__getslice__ (3, 7)[3]), 20)
        
                self.assertEqual (ar.__getslice__ (0, 0), None)
                self.assertEqual (ar.__getslice__ (5, 5), None)
                self.assertEqual (ar.__getslice__ (9, 9), None)
        
            # Has to resolve to ar[7:8]
            self.assertEqual (len (ar[-3:-2]), 20)

            # Try assignments.

            # 2D assignment.
            if PY3:
                ar[2:5] = (255, 255, 255)
            else:
                ar.__setslice__ (2, 5, (255, 255, 255))
            self.assertEqual (ar[3][3], sf.map_rgb ((255, 255, 255)))

            # 1D assignment
            if PY3:
                ar[3][3:7] = (10, 10, 10)
            else:
                ar[3].__setslice__ (3, 7, (10, 10, 10))
                
            self.assertEqual (ar[3][5], sf.map_rgb ((10, 10, 10)))
            self.assertEqual (ar[3][6], sf.map_rgb ((10, 10, 10)))

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
            sf = pygame.Surface ((6, 8), 0, bpp)
            sf.fill ((0, 0, 0))
            ar = pygame.PixelArray (sf)

            # Test single value assignment
            val = sf.map_rgb ((128, 128, 128))
            if PY3:
                ar[0:2] = val
            else:
                ar.__setslice__ (0, 2, val)
            self.assertEqual (ar[0][0], val)
            self.assertEqual (ar[0][1], val)
            self.assertEqual (ar[1][0], val)
            self.assertEqual (ar[1][1], val)

            val = sf.map_rgb ((0, 255, 255))
            ar[-3:-1] = val
            self.assertEqual (ar[3][0], val)
            self.assertEqual (ar[-2][1], val)

            val = sf.map_rgb ((255, 255, 255))
            ar[-3:] = (255, 255, 255)
            self.assertEqual (ar[4][0], val)
            self.assertEqual (ar[-1][1], val)

            # Test list assignment, this is a vertical assignment.
            val = sf.map_rgb ((0, 255, 0))
            if PY3:
                ar[2:4] = [val] * 8
            else:
                ar.__setslice__ (2, 4, [val] * 8)
            self.assertEqual (ar[2][0], val)
            self.assertEqual (ar[2][1], val)
            self.assertEqual (ar[2][4], val)
            self.assertEqual (ar[2][5], val)
            self.assertEqual (ar[3][0], val)
            self.assertEqual (ar[3][1], val)
            self.assertEqual (ar[3][4], val)
            self.assertEqual (ar[3][5], val)

            # And the horizontal assignment.
            val = sf.map_rgb ((255, 0, 0))
            val2 = sf.map_rgb ((128, 0, 255))
            if PY3:
                ar[0:2] = [val, val2]
            else:
                ar.__setslice__ (0, 2, [val, val2])
            self.assertEqual (ar[0][0], val)
            self.assertEqual (ar[1][0], val2)
            self.assertEqual (ar[0][1], val)
            self.assertEqual (ar[1][1], val2)
            self.assertEqual (ar[0][4], val)
            self.assertEqual (ar[1][4], val2)
            self.assertEqual (ar[0][5], val)
            self.assertEqual (ar[1][5], val2)

            # Test pixelarray assignment.
            ar[:] = (0, 0, 0)
            sf2 = pygame.Surface ((6, 8), 0, bpp)
            sf2.fill ((255, 0, 255))

            val = sf.map_rgb ((255, 0, 255))
            ar2 = pygame.PixelArray (sf2)

            ar[:] = ar2[:]
            self.assertEqual (ar[0][0], val)
            self.assertEqual (ar[5][7], val)

    def test_subscript (self):
        # By default we do not need to work with any special __***__
        # methods as map subscripts are the first looked up by the
        # object system.
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((6, 8), 0, bpp)
            sf.set_at ((1, 3), (0, 255, 0))
            sf.set_at ((0, 0), (0, 255, 0))
            sf.set_at ((4, 4), (0, 255, 0))
            val = sf.map_rgb ((0, 255, 0))

            ar = pygame.PixelArray (sf)

            # Test single value requests.
            self.assertEqual (ar[1,3], val)
            self.assertEqual (ar[0,0], val)
            self.assertEqual (ar[4,4], val)
            self.assertEqual (ar[1][3], val)
            self.assertEqual (ar[0][0], val)
            self.assertEqual (ar[4][4], val)

            # Test ellipse working.
            self.assertEqual (len (ar[...,...]), 6)
            self.assertEqual (len (ar[1,...]), 8)
            self.assertEqual (len (ar[...,3]), 6)

            # Test simple slicing
            self.assertEqual (len (ar[:,:]), 6)
            self.assertEqual (len (ar[:,]), 6)
            self.assertEqual (len (ar[1,:]), 8)
            self.assertEqual (len (ar[:,2]), 6)
            # Empty slices
            self.assertEqual (ar[4:4,], None)
            self.assertEqual (ar[4:4,...], None)
            self.assertEqual (ar[4:4,2:2], None)
            self.assertEqual (ar[4:4,1:4], None)
            self.assertEqual (ar[4:4:2,], None)
            self.assertEqual (ar[4:4:-2,], None)
            self.assertEqual (ar[4:4:1,...], None)
            self.assertEqual (ar[4:4:-1,...], None)
            self.assertEqual (ar[4:4:1,2:2], None)
            self.assertEqual (ar[4:4:-1,1:4], None)
            self.assertEqual (ar[...,4:4], None)
            self.assertEqual (ar[1:4,4:4], None)
            self.assertEqual (ar[...,4:4:1], None)
            self.assertEqual (ar[...,4:4:-1], None)
            self.assertEqual (ar[2:2,4:4:1], None)
            self.assertEqual (ar[1:4,4:4:-1], None)

            # Test advanced slicing
            ar[0] = 0
            ar[1] = 1
            ar[2] = 2
            ar[3] = 3
            ar[4] = 4
            ar[5] = 5

            # We should receive something like [0,2,4]
            self.assertEqual (ar[::2,1][0], 0)
            self.assertEqual (ar[::2,1][1], 2)
            self.assertEqual (ar[::2,1][2], 4)
            # We should receive something like [2,2,2]
            self.assertEqual (ar[2,::2][0], 2)
            self.assertEqual (ar[2,::2][1], 2)
            self.assertEqual (ar[2,::2][2], 2)
            
            # Should create a 3x3 array of [0,2,4]
            ar2 = ar[::2,::2]
            self.assertEqual (len (ar2), 3)
            self.assertEqual (ar2[0][0], 0)
            self.assertEqual (ar2[0][1], 0)
            self.assertEqual (ar2[0][2], 0)
            self.assertEqual (ar2[2][0], 4)
            self.assertEqual (ar2[2][1], 4)
            self.assertEqual (ar2[2][2], 4)
            self.assertEqual (ar2[1][0], 2)
            self.assertEqual (ar2[2][0], 4)
            self.assertEqual (ar2[1][1], 2)

            # Should create a reversed 3x8 array over X of [1,2,3] -> [3,2,1]
            ar2 = ar[3:0:-1]
            self.assertEqual (len (ar2), 3)
            self.assertEqual (ar2[0][0], 3)
            self.assertEqual (ar2[0][1], 3)
            self.assertEqual (ar2[0][2], 3)
            self.assertEqual (ar2[0][7], 3)
            self.assertEqual (ar2[2][0], 1)
            self.assertEqual (ar2[2][1], 1)
            self.assertEqual (ar2[2][2], 1)
            self.assertEqual (ar2[2][7], 1)
            self.assertEqual (ar2[1][0], 2)
            self.assertEqual (ar2[1][1], 2)
            # Should completely reverse the array over X -> [5,4,3,2,1,0]
            ar2 = ar[::-1]
            self.assertEqual (len (ar2), 6)
            self.assertEqual (ar2[0][0], 5)
            self.assertEqual (ar2[0][1], 5)
            self.assertEqual (ar2[0][3], 5)
            self.assertEqual (ar2[0][-1], 5)
            self.assertEqual (ar2[1][0], 4)
            self.assertEqual (ar2[1][1], 4)
            self.assertEqual (ar2[1][3], 4)
            self.assertEqual (ar2[1][-1], 4)
            self.assertEqual (ar2[-1][-1], 0)
            self.assertEqual (ar2[-2][-2], 1)
            self.assertEqual (ar2[-3][-1], 2)

            # Test advanced slicing
            ar[:] = 0
            ar2 = ar[:,1]
            ar2[:] = [99] * len(ar2)
            self.assertEqual (ar2[0], 99)
            self.assertEqual (ar2[-1], 99)
            self.assertEqual (ar2[-2], 99)
            self.assertEqual (ar2[2], 99)
            self.assertEqual (ar[0,1], 99)
            self.assertEqual (ar[1,1], 99)
            self.assertEqual (ar[2,1], 99)
            self.assertEqual (ar[-1,1], 99)
            self.assertEqual (ar[-2,1], 99)

    def test_ass_subscript (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((6, 8), 0, bpp)
            sf.fill ((255, 255, 255))
            ar = pygame.PixelArray (sf)

            # Test ellipse working
            ar[...,...] = (0, 0, 0)
            self.assertEqual (ar[0,0], 0)
            self.assertEqual (ar[1,0], 0)
            self.assertEqual (ar[-1,-1], 0)
            ar[...,] = (0, 0, 255)
            self.assertEqual (ar[0,0], sf.map_rgb ((0, 0, 255)))
            self.assertEqual (ar[1,0], sf.map_rgb ((0, 0, 255)))
            self.assertEqual (ar[-1,-1], sf.map_rgb ((0, 0, 255)))
            ar[:,...] = (255, 0, 0)
            self.assertEqual (ar[0,0], sf.map_rgb ((255, 0, 0)))
            self.assertEqual (ar[1,0], sf.map_rgb ((255, 0, 0)))
            self.assertEqual (ar[-1,-1], sf.map_rgb ((255, 0, 0)))

    def test_make_surface (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 20), 0, bpp)
            sf.fill ((255, 255, 255))
            ar = pygame.PixelArray (sf)
            newsf = ar[::2,::2].make_surface ()
            rect = newsf.get_rect ()
            self.assertEqual (rect.width, 5)
            self.assertEqual (rect.height, 10)

        # Bug when array width is not a multiple of the slice step.
        w = 17
        lst = list(range(w))
        w_slice = len(lst[::2])
        h = 3
        sf = pygame.Surface ((w, h), 0, 32)
        ar = pygame.PixelArray (sf)
        ar2 = ar[::2,:]
        sf2 = ar2.make_surface ()
        w2, h2 = sf2.get_size ()
        self.assertEqual (w2, w_slice)
        self.assertEqual (h2, h)

        # Bug when array height is not a multiple of the slice step.
        # This can hang the Python interpreter.
        h = 17
        lst = list(range(h))
        h_slice = len(lst[::2])
        w = 3
        sf = pygame.Surface ((w, h), 0, 32)
        ar = pygame.PixelArray (sf)
        ar2 = ar[:,::2]
        sf2 = ar2.make_surface ()  # Hangs here.
        w2, h2 = sf2.get_size ()
        self.assertEqual (w2, w)
        self.assertEqual (h2, h_slice)

    def test_iter (self):
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((5, 10), 0, bpp)
            ar = pygame.PixelArray (sf)
            iterations = 0
            for col in ar:
                self.assertEqual (len (col), 10)
                iterations += 1
            self.assertEqual (iterations, 5)

    def test_replace (self):
        #print "replace start"
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 10), 0, bpp)
            sf.fill ((255, 0, 0))
            rval = sf.map_rgb ((0, 0, 255))
            oval = sf.map_rgb ((255, 0, 0))
            ar = pygame.PixelArray (sf)
            ar[::2].replace ((255, 0, 0), (0, 0, 255))
            self.assertEqual (ar[0][0], rval)
            self.assertEqual (ar[1][0], oval)
            self.assertEqual (ar[2][3], rval)
            self.assertEqual (ar[3][6], oval)
            self.assertEqual (ar[8][9], rval)
            self.assertEqual (ar[9][9], oval)
            
            ar[::2].replace ((0, 0, 255), (255, 0, 0), weights=(10, 20, 50))
            self.assertEqual (ar[0][0], oval)
            self.assertEqual (ar[2][3], oval)
            self.assertEqual (ar[3][6], oval)
            self.assertEqual (ar[8][9], oval)
            self.assertEqual (ar[9][9], oval)
        #print "replace end"

    def test_extract (self):
        #print "extract start"
        for bpp in (8, 16, 24, 32):
            sf = pygame.Surface ((10, 10), 0, bpp)
            sf.fill ((0, 0, 255))
            sf.fill ((255, 0, 0), (2, 2, 6, 6))

            white = sf.map_rgb ((255, 255, 255))
            black = sf.map_rgb ((0, 0, 0))

            ar = pygame.PixelArray (sf)
            newar = ar.extract ((255, 0, 0))

            self.assertEqual (newar[0][0], black)
            self.assertEqual (newar[1][0], black)
            self.assertEqual (newar[2][3], white)
            self.assertEqual (newar[3][6], white)
            self.assertEqual (newar[8][9], black)
            self.assertEqual (newar[9][9], black)

            newar = ar.extract ((255, 0, 0), weights=(10, 0.1, 50))
            self.assertEqual (newar[0][0], black)
            self.assertEqual (newar[1][0], black)
            self.assertEqual (newar[2][3], white)
            self.assertEqual (newar[3][6], white)
            self.assertEqual (newar[8][9], black)
            self.assertEqual (newar[9][9], black)
        #print "extract end"

    def todo_test_surface(self):

        # __doc__ (as of 2008-08-02) for pygame.pixelarray.PixelArray.surface:

          # PixelArray.surface: Return Surface
          # Gets the Surface the PixelArray uses.
          # 
          # The Surface, the PixelArray was created for. 

        self.fail() 

if __name__ == '__main__':
    unittest.main()
