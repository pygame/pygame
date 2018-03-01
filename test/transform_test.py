if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

import unittest
if is_pygame_pkg:
    from pygame.tests import test_utils
else:
    from test import test_utils
import pygame
import pygame.transform
from pygame.locals import *

import platform

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
    #class TransformModuleTest( object ):

    #def assertEqual(self, x,x2):
    #    print x,x2

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


    def test_threshold_inverse_set(self):
        """ changes the pixels within the threshold, and outside.
        """
        _surf = pygame.Surface((32, 32), pygame.SRCALPHA, 32)

        dest_surf = _surf                  # surface we are changing.
        surf = _surf                       # surface we are looking at
        search_color = (55, 55, 55, 255)   # color we are searching for.
        threshold = (0, 0, 0, 0)           # within this distance from search_color.
        set_color = (5, 5, 5, 255)         # color we set.
        set_behavior = 1                   # pixels in dest_surface will be changed to color.
        search_surf = None                 # we are not comparing colors against a second surface.
        inverse_set = True                 # pixels within threshold are changed to 'set_color'

        # fill the surface with colors we are not looking for.
        surf.fill((10, 10, 10, 255))
        # set 2 pixels to the color we are searching for.
        surf.set_at((0, 0), set_color)
        surf.set_at((12, 5), set_color)

        num_threshold_pixels = pygame.transform.threshold(
            surf,
            dest_surf,
            search_color,
            threshold,
            set_color,
            set_behavior,
            search_surf,
            inverse_set)

        self.assertEqual(num_threshold_pixels, 2)
        # only two pixels changed to diff_color.
        self.assertEqual(surface.get_at((0, 0)), set_color)
        self.assertEqual(surface.get_at((12, 5)), set_color)

        # other pixels should be the same as they were before.
        self.assertEqual(surface.get_at((2, 2)), (10, 10, 10, 255))



#XXX
    def test_threshold_non_src_alpha(self):

        result  = pygame.Surface((10,10))
        s1 = pygame.Surface((10,10))
        s2 = pygame.Surface((10,10))
        s3 = pygame.Surface((10,10))
        s4 = pygame.Surface((10,10))
        result = pygame.Surface((10,10))
        x = s1.fill((0,0,0))
        x = s2.fill((0,20,0))
        x = s3.fill((0,0,0))
        x = s4.fill((0,0,0))
        s1.set_at((0,0), (32, 20, 0 ))
        s2.set_at((0,0), (33, 21, 0 ))
        s2.set_at((3,0), (63, 61, 0 ))
        s3.set_at((0,0), (112, 31, 0 ))
        s4.set_at((0,0), (11, 31, 0 ))
        s4.set_at((1,1), (12, 31, 0 ))

        self.assertEqual( s1.get_at((0,0)), (32, 20, 0, 255) )
        self.assertEqual( s2.get_at((0,0)), (33, 21, 0, 255) )
        self.assertEqual( (0,0), (s1.get_flags(), s2.get_flags()))



        #All one hundred of the pixels should be within the threshold.

        #>>> object_tracking.diff_image(result, s1, s2, threshold = 20)
        #100

        similar_color = (255, 255, 255,255)
        diff_color=(222,0,0,255)
        threshold_color = (20,20,20,255)

        rr = pygame.transform.threshold(result, s1, similar_color, threshold_color, diff_color, 1, s2)
        self.assertEqual(rr, 99)

        self.assertEqual( result.get_at((0,0)), (255,255,255, 255) )



        rr = pygame.transform.threshold(result, s1, similar_color,
                threshold_color, diff_color, 2, s2)
        self.assertEqual(rr, 99)

        self.assertEqual( result.get_at((0,0)), (32, 20, 0, 255) )




        # this is within the threshold,
        #     so the color is copied from the s1 surface.
        self.assertEqual( result.get_at((1,0)), (0, 0, 0, 255) )

        # this color was not in the threshold so it has been set to diff_color
        self.assertEqual( result.get_at((3,0)), (222, 0, 0, 255) )








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
            threshold       = threshold_template[:]

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

        s1 = pygame.Surface((32,32), SRCALPHA, 32)
        s2 = pygame.Surface((32,32), SRCALPHA, 32)
        s3 = pygame.Surface((1,1), SRCALPHA, 32)

        s1.fill((40,40,40))
        s2.fill((255,255,255))




        dest_surface = s2
        surface1 = s1
        color = (30,30,30)
        the_threshold = (11,11,11)
        diff_color = (255,0,0)
        change_return = 2

        # set the similar pixels in destination surface to the color
        #     in the first surface.
        num_threshold_pixels = threshold(dest_surface, surface1, color,
                                         the_threshold, diff_color,
                                         change_return)

        #num_threshold_pixels = threshold(s2, s1, (30,30,30))
        self.assertEqual(num_threshold_pixels, s1.get_height() * s1.get_width())
        self.assertEqual(s2.get_at((0,0)), (40, 40, 40, 255))





        if 1:

            # only one pixel should not be changed.
            s1.fill((40,40,40))
            s2.fill((255,255,255))
            s1.set_at( (0,0), (170, 170, 170) )
            # set the similar pixels in destination surface to the color
            #     in the first surface.
            num_threshold_pixels = threshold(s2, s1, (30,30,30), (11,11,11),
                                             (0,0,0), 2)

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


        if 1:
            # test end markers.  0, and 255

            # the pixels are different by 1.
            s1.fill((254,254,254))
            s2.fill((255,255,255))
            s3.fill((255,255,255))
            s1.set_at( (0,0), (170, 170, 170) )
            num_threshold_pixels = threshold(s3, s1, (254,254,254), (1,1,1),
                                             (44,44,44,255), False)
            self.assertEqual(num_threshold_pixels, (s1.get_height() * s1.get_width()) -1)


            # compare the two surfaces.  Should be all but one matching.
            num_threshold_pixels = threshold(s3, s1, 0, (1,1,1),
                                             (44,44,44,255), False, s2)
            self.assertEqual(num_threshold_pixels, (s1.get_height() * s1.get_width()) -1)


            # within (0,0,0) threshold?  Should match no pixels.
            num_threshold_pixels = threshold(s3, s1, (253,253,253), (0,0,0),
                                             (44,44,44,255), False)
            self.assertEqual(num_threshold_pixels, 0)


            # other surface within (0,0,0) threshold?  Should match no pixels.
            num_threshold_pixels = threshold(s3, s1, 0, (0,0,0),
                                             (44,44,44,255), False, s2)
            self.assertEqual(num_threshold_pixels, 0)




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



    def test_average_surfaces__24(self):

        SIZE = 32
        depth = 24
        s1 = pygame.Surface((SIZE, SIZE), 0, depth)
        s2 = pygame.Surface((SIZE, SIZE), 0, depth)
        s3 = pygame.Surface((SIZE, SIZE), 0, depth)
        s1.fill((10,10,70, 255))
        s2.fill((10,20,70, 255))
        s3.fill((10,130,10, 255))

        surfaces = [s1, s2, s3]
        sr = pygame.transform.average_surfaces(surfaces)
        self.assertEqual( sr.get_masks(), s1.get_masks() )
        self.assertEqual( sr.get_flags(), s1.get_flags() )
        self.assertEqual( sr.get_losses(), s1.get_losses() )

        if 0:
            print ( sr, s1 )
            print ( sr.get_masks(), s1.get_masks() )
            print ( sr.get_flags(), s1.get_flags() )
            print ( sr.get_losses(), s1.get_losses() )
            print ( sr.get_shifts(), s1.get_shifts() )

        self.assertEqual(sr.get_at((0,0)), (10,53,50,255))









    def test_average_color(self):
        """
        """

        a = [24, 32]
        for i in a:
            s = pygame.Surface((32,32), 0, i)
            s.fill((0,100,200))
            s.fill((10,50,100), (0,0,16,32))

            self.assertEqual(pygame.transform.average_color(s),(5,75,150,0))
            self.assertEqual(pygame.transform.average_color(s, (16,0,16,32)), (0,100,200,0))

    def todo_test_rotate(self):

        # __doc__ (as of 2008-06-25) for pygame.transform.rotate:

          # pygame.transform.rotate(Surface, angle): return Surface
          # rotate an image

        # color = (128, 128, 128, 255)
        # s = pygame.Surface((3, 3))

        # s.set_at((2, 0), color)

        # self.assert_(s.get_at((0, 0)) != color)
        # s = pygame.transform.rotate(s, 90)
        # self.assert_(s.get_at((0, 0)) == color)

        self.fail()

    def test_rotate__lossless_at_90_degrees(self):
        w, h = 32, 32
        s = pygame.Surface((w, h), pygame.SRCALPHA)

        gradient = list(test_utils.gradient(w, h))

        for pt, color in gradient: s.set_at(pt, color)

        for rotation in (90, -90):
            s = pygame.transform.rotate(s,rotation)

        for pt, color in gradient:
            self.assert_(s.get_at(pt) == color)

    def test_scale2x(self):

        # __doc__ (as of 2008-06-25) for pygame.transform.scale2x:

          # pygame.transform.scale2x(Surface, DestSurface = None): Surface
          # specialized image doubler

        w, h = 32, 32
        s = pygame.Surface((w, h), pygame.SRCALPHA, 32)

        # s.set_at((0,0), (20, 20, 20, 255))

        s2 = pygame.transform.scale2x(s)
        self.assertEquals(s2.get_rect().size, (64, 64))

    def test_get_smoothscale_backend(self):
        filter_type = pygame.transform.get_smoothscale_backend()
        self.failUnless(filter_type in ['GENERIC', 'MMX', 'SSE'])
        # It would be nice to test if a non-generic type corresponds to an x86
        # processor. But there is no simple test for this. platform.machine()
        # returns process version specific information, like 'i686'.

    def test_set_smoothscale_backend(self):
        # All machines should allow 'GENERIC'.
        original_type = pygame.transform.get_smoothscale_backend()
        pygame.transform.set_smoothscale_backend('GENERIC')
        filter_type = pygame.transform.get_smoothscale_backend()
        self.failUnlessEqual(filter_type, 'GENERIC')
        # All machines should allow returning to original value.
        # Also check that keyword argument works.
        pygame.transform.set_smoothscale_backend(type=original_type)
        # Something invalid.
        def change():
            pygame.transform.set_smoothscale_backend('mmx')
        self.failUnlessRaises(ValueError, change)
        # Invalid argument keyword.
        def change():
            pygame.transform.set_smoothscale_backend(t='GENERIC')
        self.failUnlessRaises(TypeError, change)
        # Invalid argument type.
        def change():
            pygame.transform.set_smoothscale_backend(1)
        self.failUnlessRaises(TypeError, change)
        # Unsupported type, if possible.
        if original_type != 'SSE':
            def change():
                pygame.transform.set_smoothscale_backend('SSE')
            self.failUnlessRaises(ValueError, change)
        # Should be back where we started.
        filter_type = pygame.transform.get_smoothscale_backend()
        self.failUnlessEqual(filter_type, original_type)

    def todo_test_chop(self):

        # __doc__ (as of 2008-08-02) for pygame.transform.chop:

          # pygame.transform.chop(Surface, rect): return Surface
          # gets a copy of an image with an interior area removed
          #
          # Extracts a portion of an image. All vertical and horizontal pixels
          # surrounding the given rectangle area are removed. The corner areas
          # (diagonal to the rect) are then brought together. (The original
          # image is not altered by this operation.)
          #
          # NOTE: If you want a "crop" that returns the part of an image within
          # a rect, you can blit with a rect to a new surface or copy a
          # subsurface.

        self.fail()

    def todo_test_flip(self):

        # __doc__ (as of 2008-08-02) for pygame.transform.flip:

          # pygame.transform.flip(Surface, xbool, ybool): return Surface
          # flip vertically and horizontally
          #
          # This can flip a Surface either vertically, horizontally, or both.
          # Flipping a Surface is nondestructive and returns a new Surface with
          # the same dimensions.

        self.fail()

    def todo_test_rotozoom(self):

        # __doc__ (as of 2008-08-02) for pygame.transform.rotozoom:

          # pygame.transform.rotozoom(Surface, angle, scale): return Surface
          # filtered scale and rotation
          #
          # This is a combined scale and rotation transform. The resulting
          # Surface will be a filtered 32-bit Surface. The scale argument is a
          # floating point value that will be multiplied by the current
          # resolution. The angle argument is a floating point value that
          # represents the counterclockwise degrees to rotate. A negative
          # rotation angle will rotate clockwise.

        self.fail()

    def todo_test_smoothscale(self):
        # __doc__ (as of 2008-08-02) for pygame.transform.smoothscale:

          # pygame.transform.smoothscale(Surface, (width, height), DestSurface =
          # None): return Surface
          #
          # scale a surface to an arbitrary size smoothly
          #
          # Uses one of two different algorithms for scaling each dimension of
          # the input surface as required.  For shrinkage, the output pixels are
          # area averages of the colors they cover.  For expansion, a bilinear
          # filter is used. For the amd64 and i686 architectures, optimized MMX
          # routines are included and will run much faster than other machine
          # types. The size is a 2 number sequence for (width, height). This
          # function only works for 24-bit or 32-bit surfaces.  An exception
          # will be thrown if the input surface bit depth is less than 24.
          #
          # New in pygame 1.8

        self.fail()

if __name__ == '__main__':
    #tt = TransformModuleTest()
    #tt.test_threshold_non_src_alpha()

    unittest.main()
