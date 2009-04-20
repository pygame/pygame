try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import pygame2
import pygame2.mask
from pygame2.mask import Mask
import pygame2.sdl.video as video
import pygame2.sdl.constants as sdlconst

class MaskTest (unittest.TestCase):
    def todo_test_pygame2_mask_Mask_angle(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.angle:

        # Gets the orientation of the pixels. Finds the approximate
        # orientation of the pixels in the image from -90 to 90 degrees. This
        # works best if performed on one connected component of pixels. It
        # will return 0.0 on an empty Mask.

        self.fail() 

    def todo_test_pygame2_mask_Mask_centroid(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.centroid:

        # Gets the centroid, the center of pixel mass, of the pixels in a
        # Mask. Returns a coordinate tuple for the centroid of the Mask. if
        # the Mask is empty, it will return (0,0).

        self.fail() 

    def todo_test_pygame2_mask_Mask_clear(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.clear:

        # Mask.clear () -> None
        # 
        # Clears all bits in the Mask.
        # 
        # Resets the state of all bits in the Mask to 0..

        self.fail() 

    def todo_test_pygame2_mask_Mask_connected_component(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.connected_component:

        # Mask.connected_component (x=None, y=None) -> Mask
        # 
        # Returns a Mask of a connected region of pixels.
        # 
        # Mask.connected_component (x=None, y=None) -> Mask  Returns a Mask of
        # a connected region of pixels.  This uses the SAUF algorithm to find
        # a connected component in the Mask. It checks 8 point connectivity.
        # By default, it will return the largest connected component in the
        # image. Optionally, a coordinate pair of a pixel can be specified,
        # and the connected component containing it will be returned. In the
        # event the pixel at that location is not set, the returned Mask will
        # be empty. The Mask returned is the same size as the original Mask.

        self.fail() 

    def test_pygame2_mask_Mask_connected_components(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.connected_components:

        # Mask.connected_component (x=None, y=None) -> Mask
        # 
        # Returns a Mask of a connected region of pixels.
        # 
        # Mask.connected_component (x=None, y=None) -> Mask  Returns a Mask of
        # a connected region of pixels.  This uses the SAUF algorithm to find
        # a connected component in the Mask. It checks 8 point connectivity.
        # By default, it will return the largest connected component in the
        # image. Optionally, a coordinate pair of a pixel can be specified,
        # and the connected component containing it will be returned. In the
        # event the pixel at that location is not set, the returned Mask will
        # be empty. The Mask returned is the same size as the original Mask.
        m = Mask(10,10)
        self.assertEquals(repr(m.connected_components()), "[]")
        
        comp = m.connected_component()
        self.assertEquals(m.count, comp.count)
        
        m.set_at(0,0, 1)
        m.set_at(1,1, 1)
        comp = m.connected_component()
        comps = m.connected_components()
        comps1 = m.connected_components(1)
        comps2 = m.connected_components(2)
        comps3 = m.connected_components(3)
        self.assertEquals(comp.count, comps[0].count)
        self.assertEquals(comps1[0].count, 2)
        self.assertEquals(comps2[0].count, 2)
        self.assertEquals(repr(comps3), "[]")
        
        m.set_at(9, 9, 1)
        comp = m.connected_component()
        comp1 = m.connected_component(1, 1)
        comp2 = m.connected_component(2, 2)
        comps = m.connected_components()
        comps1 = m.connected_components(1)
        comps2 = m.connected_components(2)
        comps3 = m.connected_components(3)
        self.assertEquals(comp.count, 2)
        self.assertEquals(comp1.count, 2)
        self.assertEquals(comp2.count, 0)
        self.assertEquals(len(comps), 2)
        self.assertEquals(len(comps1), 2)
        self.assertEquals(len(comps2), 1)
        self.assertEquals(len(comps3), 0)

    def todo_test_pygame2_mask_Mask_count(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.count:

        # Gets the amount of bits in the Mask.

        self.fail() 

    def test_pygame2_mask_Mask_draw(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.draw:

        # Mask.draw (mask, x, y) -> None
        # 
        # Draws the passed Mask onto the Mask.
        # 
        # Mask.draw (mask, x, y) -> None  Draws the passed Mask onto the Mask.
        # This performs a bitwise OR operation upon the calling Mask. The
        # passed mask's start offset for the draw operation will be the x and
        # y offset passed to the method.
        m = Mask (100, 100)
        self.assertEqual(m.count, 0)
        
        m.fill()
        self.assertEqual(m.count, 10000)
        
        m2 = Mask (10,10)
        m2.fill()
        m.erase (m2, (50, 50))
        self.assertEqual(m.count, 9900)
        
        m.invert()
        self.assertEqual(m.count, 100)
        
        m.draw(m2, (0,0))
        self.assertEqual(m.count, 200)    
        
        m.clear()
        self.assertEqual(m.count, 0)

    def todo_test_pygame2_mask_Mask_erase(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.erase:

        # Mask.erase (mask, x, y) -> None
        # 
        # Erases the passed Mask from the Mask.
        # 
        # Mask.erase (mask, x, y) -> None  Erases the passed Mask from the
        # Mask.  This performs a bitwise NAND operation upon the calling Mask.
        # The passed mask's start offfset for the erase operation will be the
        # x and y offset passed to the method.

        self.fail() 

    def todo_test_pygame2_mask_Mask_fill(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.fill:

        # Mask.fill () -> None
        # 
        # Sets all bits to 1 within the Mask.

        self.fail() 

    def todo_test_pygame2_mask_Mask_get_at(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.get_at:

        # Mask.get_at (x, y) -> int
        # 
        # Gets the bit value at the desired location.

        self.fail() 

    def test_pygame2_mask_Mask_get_bounding_rects(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.get_bounding_rects:

        # Mask.get_bounding_rects () -> [Mask, Mask ...]
        # 
        # Returns a list of bounding rects of regions of set pixels.
        # 
        # Mask.get_bounding_rects () -> [Mask, Mask ...]  Returns a list of
        # bounding rects of regions of set pixels.  This gets a bounding rect
        # of connected regions of set bits. A bounding rect is one for which
        # each of the connected pixels is inside the rect.
        m = Mask (10, 10)
        m.set_at(0, 0, 1)
        m.set_at(1, 0, 1)

        m.set_at(0, 1, 1)

        m.set_at(0,3, 1)
        m.set_at(3,3, 1)
        
        r = m.get_bounding_rects()

        self.assertEquals(repr(r), "[(0, 0, 2, 2), (0, 3, 1, 1), (3, 3, 1, 1)]")

        #1100
        #1111
        m = Mask(4,2)
        m.set_at(0,0, 1)
        m.set_at(1,0, 1)
        m.set_at(2,0, 0)
        m.set_at(3,0, 0)

        m.set_at(0,1, 1)
        m.set_at(1,1, 1)
        m.set_at(2,1, 1)
        m.set_at(3,1, 1)
 
        r = m.get_bounding_rects()
        self.assertEquals(repr(r), "[(0, 0, 4, 2)]")

        #00100
        #01110
        #00100
        m = Mask(5,3)
        m.set_at(0,0, 0)
        m.set_at(1,0, 0)
        m.set_at(2,0, 1)
        m.set_at(3,0, 0)
        m.set_at(4,0, 0)

        m.set_at(0,1, 0)
        m.set_at(1,1, 1)
        m.set_at(2,1, 1)
        m.set_at(3,1, 1)
        m.set_at(4,1, 0)

        m.set_at(0,2, 0)
        m.set_at(1,2, 0)
        m.set_at(2,2, 1)
        m.set_at(3,2, 0)
        m.set_at(4,2, 0)

        r = m.get_bounding_rects()
        self.assertEquals(repr(r), "[(1, 0, 3, 3)]")

        #00010
        #00100
        #01000
        m = Mask(5,3)
        m.set_at(0,0, 0)
        m.set_at(1,0, 0)
        m.set_at(2,0, 0)
        m.set_at(3,0, 1)
        m.set_at(4,0, 0)

        m.set_at(0,1, 0)
        m.set_at(1,1, 0)
        m.set_at(2,1, 1)
        m.set_at(3,1, 0)
        m.set_at(4,1, 0)

        m.set_at(0,2, 0)
        m.set_at(1,2, 1)
        m.set_at(2,2, 0)
        m.set_at(3,2, 0)
        m.set_at(4,2, 0)

        r = m.get_bounding_rects()
        self.assertEquals(repr(r), "[(1, 0, 3, 3)]")

        #00011
        #11111
        m = Mask(5,2)
        m.set_at(0,0, 0)
        m.set_at(1,0, 0)
        m.set_at(2,0, 0)
        m.set_at(3,0, 1)
        m.set_at(4,0, 1)

        m.set_at(0,1, 1)
        m.set_at(1,1, 1)
        m.set_at(2,1, 1)
        m.set_at(3,1, 1)
        m.set_at(3,1, 1)
 
        r = m.get_bounding_rects()
        self.assertEquals(repr(r), "[(0, 0, 5, 2)]")

    def todo_test_pygame2_mask_Mask_height(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.height:

        # Gets the height of the Mask.

        self.fail() 

    def todo_test_pygame2_mask_Mask_invert(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.invert:

        # Mask.invert () -> None
        # 
        # Inverts all bits in the Mask.

        self.fail() 

    def test_pygame2_mask_Mask_outline(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.outline:
        m = Mask (20, 20)
        self.assertEqual(m.outline(), [])
        
        m.set_at(10, 10, 1)
        self.assertEqual(m.outline(), [(10,10)])
        
        m.set_at( 10, 12, 1)
        self.assertEqual(m.outline(10), [(10,10)])
        
        m.set_at(11, 11, 1)
        self.assertEqual(m.outline(), [(10,10), (11,11), (10,12), (11,11),
                                       (10,10)])
        self.assertEqual(m.outline(2), [(10,10), (10,12), (10,10)])

    def todo_test_pygame2_mask_Mask_overlap(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.overlap:

        # Mask.overlap (mask, x, y) -> int, int
        # 
        # Returns nonzero if the masks overlap with the given offset.
        # 
        # Mask.overlap (mask, x, y) -> int, int  Returns nonzero if the masks
        # overlap with the given offset.  The overlap tests uses the following
        # offsets (which may be negative):  +----+----------.. |A   | yoffset
        # |  +-+----------.. +--|B |xoffset |  | :  :
        # 
        # Mask.overlap (mask, x, y) -> int, int  Returns nonzero if the masks
        # overlap with the given offset.  The overlap tests uses the following
        # offsets (which may be negative):  +----+----------.. |A   | yoffset
        # |  +-+----------.. +--|B |xoffset |  | :  :

        self.fail() 

    def todo_test_pygame2_mask_Mask_overlap_area(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.overlap_area:

        # Mask.overlap_area (mask, x, y) -> int
        # 
        # Returns the number of overlapping bits of two Masks.
        # 
        # Mask.overlap_area (mask, x, y) -> int  Returns the number of
        # overlapping bits of two Masks.  This returns how many pixels overlap
        # with the other mask given. It can be used to see in which direction
        # things collide, or to see how much the two masks collide.

        self.fail() 

    def todo_test_pygame2_mask_Mask_overlap_mask(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.overlap_mask:

        # Mask.overlap_mask (mask, x, y) -> Mask.
        # 
        # Returns a mask with the overlap of two other masks. A bitwise AND.

        self.fail() 

    def todo_test_pygame2_mask_Mask_scale(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.scale:

        # Mask.scale (width, height) -> Mask
        # 
        # Creates a new scaled Mask with the given width and height.
        # 
        # Mask.scale (width, height) -> Mask  Creates a new scaled Mask with
        # the given width and height.  The quality of the scaling may not be
        # perfect for all circumstances, but it should be reasonable. If
        # either w or h is 0 a clear 1x1 mask is returned.

        self.fail() 

    def test_pygame2_mask_Mask_set_at(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.set_at:

        # Mask.set_at (x, y) -> None
        # 
        # Sets the bit value at the desired location.
        m = Mask (10, 10)
        m.set_at (0, 0, 1)
        self.assertEqual(m.get_at(0, 0), 1)
        m.set_at(9, 0, 1)
        self.assertEqual(m.get_at(9, 0), 1)

        # out of bounds, should get IndexError
        self.assertRaises(IndexError, lambda : m.get_at (-1,0) )
        self.assertRaises(IndexError, lambda : m.set_at (-1, 0, 1) )
        self.assertRaises(IndexError, lambda : m.set_at (10, 0, 1) )
        self.assertRaises(IndexError, lambda : m.set_at (0, 10, 1) )

    def todo_test_pygame2_mask_Mask_size(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.size:

        # Mask (width, height) -> Mask
        # 
        # Creates a new, empty Mask with the desired dimensions.
        # 
        # Mask (width, height) -> Mask  Creates a new, empty Mask with the
        # desired dimensions.  The Mask is a 2D array using single bits to
        # represent states within a x,y matrix. This makes it suitable for
        # pixel-perfect overlap handling of image buffers.

        self.fail() 

    def todo_test_pygame2_mask_Mask_width(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.Mask.width:

        # Gets the width of the Mask

        self.fail() 

    def test_pygame2_mask_from_surface(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.from_surface:

        # pygame2.mask.from_surface (surface, threshold) -> Mask
        # 
        # Returns a Mask from the given pygame2.sdl.video.Surface.
        # 
        # pygame2.mask.from_surface (surface, threshold) -> Mask  Returns a
        # Mask from the given pygame2.sdl.video.Surface.  Makes the
        # transparent parts of the Surface not set, and the opaque parts set.
        # The alpha of each pixel is checked to see if it is greater than the
        # given threshold. If the Surface is color keyed, then threshold is
        # not used.  This requires pygame2 to be built with SDL support
        # enabled.
        # 
        # This requires pygame2 to be built with SDL support enabled.
        video.init ()
        
        mask_from_surface = pygame2.mask.from_surface

        surf = video.Surface(70, 70, 32, sdlconst.SRCALPHA)
        surf.fill(pygame2.Color(255,255,255,255))

        amask = mask_from_surface (surf)

        self.assertEqual(amask.get_at(0,0), 1)
        self.assertEqual(amask.get_at(66,1), 1)
        self.assertEqual(amask.get_at(69,1), 1)

        surf.set_at(0,0, pygame2.Color(255,255,255,127))
        surf.set_at(1,0, pygame2.Color(255,255,255,128))
        surf.set_at(2,0, pygame2.Color(255,255,255,0))
        surf.set_at(3,0, pygame2.Color(255,255,255,255))

        amask = mask_from_surface(surf)
        self.assertEqual(amask.get_at(0,0), 0)
        self.assertEqual(amask.get_at(1,0), 1)
        self.assertEqual(amask.get_at(2,0), 0)
        self.assertEqual(amask.get_at(3,0), 1)

        surf.fill(pygame2.Color(255,255,255,0))
        amask = mask_from_surface(surf)
        self.assertEqual(amask.get_at(0, 0), 0)

    def test_pygame2_mask_from_threshold(self):

        # __doc__ (as of 2008-11-03) for pygame2.mask.from_threshold:

        # TODO
        video.init ()
        a = [16, 24, 32]
        
        for i in a:
            surf = video.Surface (70, 70, i)
            surf.fill (pygame2.Color (100,50,200), pygame2.Rect (20,20,20,20))
            mask = pygame2.mask.from_threshold (surf,
                                                pygame2.Color (100,50,200,255),
                                                pygame2.Color (10,10,10,255))
            
            self.assertEqual (mask.count, 400)
            self.assertEqual (mask.get_bounding_rects (),
                              [pygame2.Rect (20, 20, 20, 20)])
            
        for i in a:
            surf = video.Surface (70, 70, i)
            surf2 = video.Surface (70,70, i)
            surf.fill (pygame2.Color (100, 100, 100))
            surf2.fill (pygame2.Color (150, 150, 150))
            surf2.fill (pygame2.Color (100, 100, 100),
                        pygame2.Rect (40, 40, 10, 10))
            mask = pygame2.mask.from_threshold(surf, pygame2.Color (0, 0, 0, 0),
                                               pygame2.Color (10, 10, 10, 255),
                                               surf2)
            
            self.assertEqual (mask.count, 100)
            self.assertEqual (mask.get_bounding_rects(),
                              [pygame2.Rect (40, 40, 10, 10)])

if __name__ == "__main__":
    unittest.main ()
