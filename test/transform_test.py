import test_utils
import test.unittest as unittest

from test_utils import test_not_implemented

import pygame, pygame.transform
from pygame.locals import *

def show_image(s, images = []):
    #pygame.display.init()
    size = s.get_rect()[2:]
    screen = pygame.display.set_mode(size)
    screen.blit(s, (0,0))
    pygame.display.flip()
    pygame.event.pump()
    going = True
    idx = 0
    while going:
        events = pygame.event.get()
        for e in events:
            if e.type == QUIT:
                going = False
            if e.type == KEYDOWN:
                if e.key in [K_s, K_a]:
                    if e.key == K_s: idx += 1
                    if e.key == K_a: idx -= 1
                    s = images[idx]
                    screen.blit(s, (0,0))
                    pygame.display.flip()
                    pygame.event.pump()
                else:
                    going = False
    pygame.display.quit()
    pygame.display.init()

def threshold(return_surf, surf, color, threshold = (0,0,0), diff_color = (0,0,0), change_return = True ):
    """ given the color it makes return_surf only have areas with the given colour.
    """
    
    width, height =surf.get_width(), surf.get_height()

    if change_return:
        return_surf.fill(diff_color)

    try:
        r, g, b = color
    except ValueError:
        r, g, b, a = color


    try:
        tr, tg, tb = color
    except ValueError:
        tr, tg, tb, ta = color



    similar = 0
    for y in xrange(height):
        for x in xrange(width):
            c1 = surf.get_at((x,y))

            if ( (abs(c1[0] - r) < tr) &
                 (abs(c1[1] - g) < tg) & 
                 (abs(c1[2] - b) < tb) ):
                # this pixel is within the threshold.
                if change_return:
                    return_surf.set_at((x,y), c1)
                similar += 1
            #else:
            #    print c1, c2
    
    
    return similar


class TransformModuleTest( unittest.TestCase ):
    def test_scale__alpha( self ):
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


    def test_scale__destination( self ):
        """ see if the destination surface can be passed in to use.
        """

        s = pygame.Surface((32,32))
        s2 = pygame.transform.scale(s, (64,64))
        s3 = s2.copy()

        s3 = pygame.transform.scale(s, (64,64), s3)
        pygame.transform.scale(s, (64,64), s2)

        # the wrong size surface is past in.  Should raise an error.
        self.assertRaises(ValueError, pygame.transform.scale, s, (33,64), s3)

        if 1:
            s = pygame.Surface((32,32))
            s2 = pygame.transform.smoothscale(s, (64,64))
            s3 = s2.copy()

            s3 = pygame.transform.smoothscale(s, (64,64), s3)
            pygame.transform.smoothscale(s, (64,64), s2)

            # the wrong size surface is past in.  Should raise an error.
            self.assertRaises(ValueError, pygame.transform.smoothscale, s, (33,64), s3)

    
    def test_threshold__honors_third_surface(self):
        # __doc__ for threshold as of Tue 07/15/2008

        # pygame.transform.threshold(DestSurface, Surface, color, threshold =
        # (0,0,0,0), diff_color = (0,0,0,0), change_return = True, Surface =
        # None): return num_threshold_pixels

        # When given the optional third
        # surface, it would use the colors in that rather than the "color"
        # specified in the function to check against.
        
        # New in pygame 1.8

        ################################################################         
        # Sizes
        (w, h) = size = (32, 32)

        # the original_color is within the threshold of the threshold_color
        threshold = (20, 20, 20, 20)

        original_color = (25,25,25,25)
        threshold_color = (10, 10, 10, 10)

        # Surfaces
        original_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
        dest_surface    = pygame.Surface(size, pygame.SRCALPHA, 32)
        
        # Third surface is used in lieu of 3rd position arg color
        third_surface   = pygame.Surface(size, pygame.SRCALPHA, 32)

        # Color filling
        original_surface.fill(original_color)
        third_surface.fill(threshold_color)

        ################################################################
        # All pixels for color should be within threshold
        # 
        pixels_within_threshold = pygame.transform.threshold (
            dest_surface, original_surface, threshold_color,
            threshold,
            0, # diff_color
            0  # change_return
        )

        self.assertEqual(w*h, pixels_within_threshold)

        ################################################################  
        # This should respect third_surface colors in place of 3rd arg 
        # color Should be the same as: surface.fill(threshold_color)
        # all within threshold
        
        pixels_within_threshold = pygame.transform.threshold (
            dest_surface, 
            original_surface,                              
            0,                            # color (would fail if honored)
            threshold,
            0,                                              # diff_color 
            0,                                           # change_return
            third_surface,
        )
        self.assertEqual(w*h, pixels_within_threshold)
        
        
        ################################################################  
        # Change dest_surface on return (not expected)

        change_color = (255, 10, 10, 10)

        pixels_within_threshold = pygame.transform.threshold (
            dest_surface, 
            original_surface,                              
            0,                                           # color 
            threshold,
            change_color,                                # diff_color 
            1,                                           # change_return
            third_surface,
        )
        
        # Return, of pixels within threshold is correct
        self.assertEqual(w*h, pixels_within_threshold)
        
        # Size of dest surface is correct
        dest_rect = dest_surface.get_rect()
        dest_size = dest_rect.size
        self.assertEqual(size, dest_size)
        
        # The color is not the change_color specified for every pixel As all
        # pixels are within threshold

        for pt in test_utils.rect_area_pts(dest_rect):
            self.assert_(dest_surface.get_at(pt) != change_color)
        
        ################################################################  
        # Lowering the threshold, expecting changed surface        

        pixels_within_threshold = pygame.transform.threshold (
            dest_surface, 
            original_surface,                              
            0,                                           # color 
            0,                                           # threshold
            change_color,                                # diff_color 
            1,                                           # change_return
            third_surface,
        )
        
        # Return, of pixels within threshold is correct
        self.assertEqual(0, pixels_within_threshold)
        
        # Size of dest surface is correct
        dest_rect = dest_surface.get_rect()
        dest_size = dest_rect.size
        self.assertEqual(size, dest_size)
        
        # The color is the change_color specified for every pixel As all
        # pixels are not within threshold
        
        for pt in test_utils.rect_area_pts(dest_rect):
            self.assertEqual(dest_surface.get_at(pt), change_color)


    def test_threshold__uneven_colors(self):
        (w,h) = size = (16, 16)
        
        original_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
        dest_surface    = pygame.Surface(size, pygame.SRCALPHA, 32)
        
        original_surface.fill(0)

        threshold_color_template = [5, 5, 5, 5]
        threshold_template       = [6, 6, 6, 6]

        ################################################################  

        for pos in range(len('rgb')):
            threshold_color = threshold_color_template[:]
            threshold       = threshold_template

            threshold_color[pos] = 45
            threshold[pos] = 50

            pixels_within_threshold = pygame.transform.threshold (
                dest_surface, original_surface, threshold_color,
                threshold,
                0, # diff_color
                0  # change_return
            )
            
            self.assertEqual(w*h, pixels_within_threshold)
                
        ################################################################  

    def test_threshold__surface(self):
        """
        """

        #pygame.transform.threshold(DestSurface, Surface, color, threshold = (0,0,0,0), diff_color = (0,0,0,0), change_return = True): return num_threshold_pixels
        threshold = pygame.transform.threshold

        s1 = pygame.Surface((32,32))
        s2 = pygame.Surface((32,32))
        s3 = pygame.Surface((1,1))

        s1.fill((40,40,40))
        s2.fill((255,255,255))


        num_threshold_pixels = threshold(s2, s1, (30,30,30), (11,11,11), (255,0,0), True)
        #num_threshold_pixels = threshold(s2, s1, (30,30,30))
        self.assertEqual(num_threshold_pixels, s1.get_height() * s1.get_width())
        self.assertEqual(s2.get_at((0,0)), (40, 40, 40, 255))

        if 1:

            # only one pixel should not be changed.
            s1.fill((40,40,40))
            s2.fill((255,255,255))
            s1.set_at( (0,0), (170, 170, 170) )
            num_threshold_pixels = threshold(s2, s1, (30,30,30), (11,11,11), (0,0,0), True)
            #num_threshold_pixels = threshold(s2, s1, (30,30,30))
            self.assertEqual(num_threshold_pixels, (s1.get_height() * s1.get_width()) -1)
            self.assertEqual(s2.get_at((0,0)), (0,0,0, 255))
            self.assertEqual(s2.get_at((0,1)), (40, 40, 40, 255))
            self.assertEqual(s2.get_at((17,1)), (40, 40, 40, 255))


        # abs(40 - 255) < 100
        #(abs(c1[0] - r) < tr)

        if 1:
            s1.fill((160,160,160))
            s2.fill((255,255,255))
            num_threshold_pixels = threshold(s2, s1, (255,255,255), (100,100,100), (0,0,0), True)

            self.assertEqual(num_threshold_pixels, (s1.get_height() * s1.get_width()))




        if 1:
            # only one pixel should not be changed.
            s1.fill((40,40,40))
            s2.fill((255,255,255))
            s1.set_at( (0,0), (170, 170, 170) )
            num_threshold_pixels = threshold(s3, s1, (30,30,30), (11,11,11), (0,0,0), False)
            #num_threshold_pixels = threshold(s2, s1, (30,30,30))
            self.assertEqual(num_threshold_pixels, (s1.get_height() * s1.get_width()) -1)



    def test_laplacian(self):
        """
        """
 
        SIZE = 32
        s1 = pygame.Surface((SIZE, SIZE))
        s2 = pygame.Surface((SIZE, SIZE))
        s1.fill((10,10,70))
        pygame.draw.line(s1, (255,0,0), (3,10), (20,20))

        # a line at the last row of the image.
        pygame.draw.line(s1, (255,0,0), (0,31), (31,31))


        pygame.transform.laplacian(s1,s2)
        
        #show_image(s1)
        #show_image(s2)

        self.assertEqual(s2.get_at((0,0)), (0, 0, 0, 255))
        self.assertEqual(s2.get_at((3,10)), (255,0,0,255))
        self.assertEqual(s2.get_at((0,31)), (255,0,0,255))
        self.assertEqual(s2.get_at((31,31)), (255,0,0,255))
        

        # here we create the return surface.
        s2 = pygame.transform.laplacian(s1)
        
        self.assertEqual(s2.get_at((0,0)), (0, 0, 0, 255))
        self.assertEqual(s2.get_at((3,10)), (255,0,0,255))
        self.assertEqual(s2.get_at((0,31)), (255,0,0,255))
        self.assertEqual(s2.get_at((31,31)), (255,0,0,255))

    def test_average_surfaces(self):
        """
        """
 
        SIZE = 32
        s1 = pygame.Surface((SIZE, SIZE))
        s2 = pygame.Surface((SIZE, SIZE))
        s3 = pygame.Surface((SIZE, SIZE))
        s1.fill((10,10,70))
        s2.fill((10,20,70))
        s3.fill((10,130,10))

        surfaces = [s1, s2, s3]
        surfaces = [s1, s2]
        sr = pygame.transform.average_surfaces(surfaces)

        self.assertEqual(sr.get_at((0,0)), (10,15,70,255))


        self.assertRaises(TypeError, pygame.transform.average_surfaces, 1)
        self.assertRaises(TypeError, pygame.transform.average_surfaces, [])

        self.assertRaises(TypeError, pygame.transform.average_surfaces, [1])
        self.assertRaises(TypeError, pygame.transform.average_surfaces, [s1, 1])
        self.assertRaises(TypeError, pygame.transform.average_surfaces, [1, s1])
        self.assertRaises(TypeError, pygame.transform.average_surfaces, [s1, s2, 1])

        self.assertRaises(TypeError, pygame.transform.average_surfaces, (s for s in [s1, s2,s3] ))


    def test_chop(self):

        # __doc__ (as of 2008-06-25) for pygame.transform.chop:

          # pygame.transform.chop(Surface, rect): return Surface
          # gets a copy of an image with an interior area removed

        self.assert_(test_not_implemented()) 

    def test_flip(self):

        # __doc__ (as of 2008-06-25) for pygame.transform.flip:

          # pygame.transform.flip(Surface, xbool, ybool): return Surface
          # flip vertically and horizontally

        self.assert_(test_not_implemented()) 

    def test_rotate(self):
        
        # __doc__ (as of 2008-06-25) for pygame.transform.rotate:

          # pygame.transform.rotate(Surface, angle): return Surface
          # rotate an image
        
        self.assert_(test_not_implemented()) 
        
        # color = (128, 128, 128, 255)
        # s = pygame.Surface((3, 3))
        
        # s.set_at((2, 0), color)

        # self.assert_(s.get_at((0, 0)) != color)
        # s = pygame.transform.rotate(s, 90)
        # self.assert_(s.get_at((0, 0)) == color)

    def test_rotate__lossless_at_90_degrees(self):
        w, h = 32, 32
        s = pygame.Surface((w, h), pygame.SRCALPHA)

        gradient = list(test_utils.gradient(w, h))

        for pt, color in gradient: s.set_at(pt, color)

        for rotation in (90, -90):
            s = pygame.transform.rotate(s,rotation)

        for pt, color in gradient:
            self.assert_(s.get_at(pt) == color)

    def test_rotozoom(self):

        # __doc__ (as of 2008-06-25) for pygame.transform.rotozoom:

          # pygame.transform.rotozoom(Surface, angle, scale): return Surface
          # filtered scale and rotation

        self.assert_(test_not_implemented()) 

    def test_scale2x(self):

        # __doc__ (as of 2008-06-25) for pygame.transform.scale2x:

          # pygame.transform.scale2x(Surface, DestSurface = None): Surface
          # specialized image doubler

        self.assert_(test_not_implemented()) 

    def test_smoothscale(self):

        # __doc__ (as of 2008-06-25) for pygame.transform.smoothscale:

          # pygame.transform.smoothscale(Surface, (width, height), DestSurface = None): return Surface
          # scale a surface to an arbitrary size smoothly

        self.assert_(test_not_implemented()) 

if __name__ == '__main__':
    unittest.main()