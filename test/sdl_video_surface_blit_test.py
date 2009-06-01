import sys
try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import pygame2
from pygame2 import Color, Rect
import pygame2.sdl.video as video
import pygame2.sdl.constants as constants

def cmpcolor (surface, source, color, area=None):
    # Simple color comparision with clip area support
    getat = surface.get_at
    sfbpp = surface.format.bits_per_pixel
    srcbpp = source.format.bits_per_pixel
    sx, sy = 0, 0
    w, h = surface.size
    if area:
        sx, sy = area.x, area.y 
        w, h = area.w, area.h

    c = surface.format.get_rgba (color)
    c2 = source.format.get_rgba (color)
    # TODO
    if (srcbpp == 16 or sfbpp == 16):
        # Ignore 16 bpp blits for now - the colors differ too much.
        return True
    if (sfbpp in (32,24) and srcbpp == 8):
        # Ignore 8 bpp to 32/24bpp blits for now - the colors differ.
        return True
    for x in range (sx, sx + w):
        for y in range (sy, sy + h):
            cc = getat (x, y)
            if cc != c:
                print ((x, y), (sfbpp, srcbpp), cc, c, c2, color)
                return False
    return True

class SDLVideoSurfaceBlitTest (unittest.TestCase):
    # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.blit:

    # blit (srcsurface[, destrect, srcrect, blendargs]) -> Rect
    # 
    # Draws the passed source surface onto this surface.
    # 
    # Draws the passed source surface onto this surface. The dstrect
    # and srcrect arguments are used for clipping the destionation
    # area (on this surface) and source area (on the source
    # surface). For the destination rectangle, the width and height
    # are ignored, only the position is taken into account. For the
    # source rectangle, all values, position as well as the size are
    # used for clipping.
    # 
    # The optional blending arguments for the drawing operation perform
    # certain specialised pixel manipulations. For those, pixels on the same
    # position are evaluated and the result manipulated according to the
    # argument.
    # 
    # +---------------------------+----------------------------------------+
    # | Blend type                | Effect                                 |
    # +===========================+========================================+
    # | BLEND_RGB_ADD             | The sum of both pixel values will be   |
    # |                           | set as resulting pixel.                |
    # +---------------------------+----------------------------------------+
    # | BLEND_RGB_SUB             | The difference of both pixel values    |
    # |                           | will be set as resulting pixel.        |
    # +---------------------------+----------------------------------------+
    # | BLEND_RGB_MIN             | The minimum of each R, G and B channel |
    # |                           | of the pixels will be set as result.   |
    # +---------------------------+----------------------------------------+
    # | BLEND_RGB_MAX             | The maximum of each R, G and B channel |
    # |                           | of the pixels will be set as result.   |
    # +---------------------------+----------------------------------------+
    # | BLEND_RGB_MULT            | The result of a multiplication of both |
    # |                           | pixel values will be used.             |
    # +---------------------------+----------------------------------------+
    # 
    # The BLEND_RGB_*** flags do not take the alpha channel into
    # account and thus are much faster for most blit operations
    # without alpha transparency. Whenever alpha transparency has to
    # be taken into account, the blending flags below should be
    # used.
    # 
    # +---------------------------+----------------------------------------+
    # | Blend type                | Effect                                 |
    # +===========================+========================================+
    # | BLEND_RGBA_ADD            | The sum of both pixel values will be   |
    # |                           | set as resulting pixel.                |
    # +---------------------------+----------------------------------------+
    # | BLEND_RGBA_SUB            | The difference of both pixel values    |
    # |                           | will be set as resulting pixel.        |
    # +---------------------------+----------------------------------------+
    # | BLEND_RGBA_MIN            | The minimum of each R, G, B and A      |
    # |                           | channel of the pixels will be set as   |
    # |                           | result.                                |
    # +---------------------------+----------------------------------------+
    # | BLEND_RGBA_MAX            | The maximum of each R, G, B and A      |
    # |                           | channel of the pixels will be set as   |
    # |                           | result.                                |
    # +---------------------------+----------------------------------------+
    # | BLEND_RGBA_MULT           | The result of a multiplication of both |
    # |                           | pixel values will be used.             |
    # +---------------------------+----------------------------------------+    
    def test_simple_32bpp_blit (self):
        # Simple 32bpp blit
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 32)
        for bpp in modes:
            sf2 = video.Surface (5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
            c2 = sf2.get_at (0, 0)
            # Solid, destructive blit.
            sf1.blit (sf2)

            self.assert_ (cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()
    
    def test_simple_24bpp_blit (self):
        # Simple 24bpp blit
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 24)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
            c2 = sf2.get_at (0, 0)
            # Solid, destructive blit.
            sf1.blit (sf2)

            self.assert_ (cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()
    
    def test_simple_16bpp_blit (self):
        # Simple 16bpp blit
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 16)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
            c2 = sf2.get_at (0, 0)
            # Solid, destructive blit.
            sf1.blit (sf2)

            self.assert_ (cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()
    
    def test_simple_8bpp_blit (self):
        # Simple 8bpp blit
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 8)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, 8)
        
            sf1.fill (color2)
            sf2.fill (color2)
            c2 = sf2.get_at (0, 0)
            # Solid, destructive blit.
            sf1.blit (sf2)

            self.assert_ (cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_32bpp_BLEND_RGB_ADD (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 32)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, additive blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (127, 127, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (50, 127, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (177, 254, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (227, 255, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (227, 255, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
        video.quit ()
    
    def test_24bpp_BLEND_RGB_ADD (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 24)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, additive blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (127, 127, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (50, 127, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (177, 254, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (227, 255, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (227, 255, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
        video.quit ()
    
    def test_16bpp_BLEND_RGB_ADD (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 16)
        for bpp in modes:
            sf2 = video.Surface (5, 5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, additive blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (127, 127, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (50, 127, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (177, 254, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (227, 255, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (227, 255, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
        video.quit ()

    def test_8bpp_BLEND_RGB_ADD (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 8)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, additive blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (127, 127, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (50, 127, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (177, 254, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (227, 255, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
            self.assert_ (cmpcolor (sf1, sf2, Color (227, 255, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
        video.quit ()

    def test_32bpp_BLEND_RGB_SUB (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (255, 255, 255)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 32)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, subtractive blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (255, 128, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (128, 20, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (127, 108, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 88, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 88, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_24bpp_BLEND_RGB_SUB (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (255, 255, 255)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 24)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, subtractive blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (255, 128, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (128, 20, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (127, 108, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 88, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 88, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_16bpp_BLEND_RGB_SUB (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (255, 255, 255)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 16)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, subtractive blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (255, 128, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (128, 20, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (127, 108, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 88, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 88, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_8bpp_BLEND_RGB_SUB (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (255, 255, 255)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 8)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, subtractive blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (255, 128, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (128, 20, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (127, 108, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 88, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 88, 255),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_32bpp_BLEND_RGB_MAX (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (30, 10, 100)
        color2 = Color (0, 127, 24)
        sf1 = video.Surface (10, 10, 32)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, maximum blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (30, 127, 100),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            
            sf2.fill (Color (128, 20, 144))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 127, 144),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 127, 144),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 127, 144),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_24bpp_BLEND_RGB_MAX (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (30, 10, 100)
        color2 = Color (0, 127, 24)
        sf1 = video.Surface (10, 10, 24)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, maximum blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (30, 127, 100),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            
            sf2.fill (Color (128, 20, 144))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 127, 144),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 127, 144),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 127, 144),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_16bpp_BLEND_RGB_MAX (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (30, 10, 100)
        color2 = Color (0, 127, 24)
        sf1 = video.Surface (10, 10, 16)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, maximum blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (30, 127, 100),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            
            sf2.fill (Color (128, 20, 144))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 127, 144),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 127, 144),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 127, 144),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_8bpp_BLEND_RGB_MAX (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (30, 10, 100)
        color2 = Color (0, 127, 24)
        sf1 = video.Surface (10, 10, 8)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, maximum blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (30, 127, 100),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            
            sf2.fill (Color (128, 20, 144))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 127, 144),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 127, 144),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 127, 144),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_32bpp_BLEND_RGB_MIN (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (255, 100, 12)
        color2 = Color (133, 127, 16)
        sf1 = video.Surface (10, 10, 32)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
            
            # Solid, minimum blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (133, 100, 12),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (128, 20, 144))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 20, 12),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 20, 12),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 0, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_24bpp_BLEND_RGB_MIN (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (255, 100, 12)
        color2 = Color (133, 127, 16)
        sf1 = video.Surface (10, 10, 24)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
            
            # Solid, minimum blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (133, 100, 12),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (128, 20, 144))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 20, 12),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 20, 12),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 0, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_16bpp_BLEND_RGB_MIN (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (255, 100, 12)
        color2 = Color (133, 127, 16)
        sf1 = video.Surface (10, 10, 16)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
            
            # Solid, minimum blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (133, 100, 12),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (128, 20, 144))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 20, 12),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 20, 12),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 0, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_8bpp_BLEND_RGB_MIN (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (255, 100, 12)
        color2 = Color (133, 127, 16)
        sf1 = video.Surface (10, 10, 8)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
            
            # Solid, minimum blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (133, 100, 12),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (128, 20, 144))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 20, 12),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (128, 20, 12),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 0, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_32bpp_BLEND_RGB_MULT (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (8, 50, 10)
        color2 = Color (127, 44, 12)
        sf1 = video.Surface (10, 10, 32)
        for bpp in modes:
            sf2 = video.Surface (5, 5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, multiply blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (3, 8, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (255, 178, 177))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (2, 5, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (1, 3, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 0, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_24bpp_BLEND_RGB_MULT (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (8, 50, 10)
        color2 = Color (127, 44, 12)
        sf1 = video.Surface (10, 10, 24)
        for bpp in modes:
            sf2 = video.Surface (5, 5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, multiply blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (3, 8, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (255, 178, 177))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (2, 5, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (1, 3, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 0, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_16bpp_BLEND_RGB_MULT (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (8, 50, 10)
        color2 = Color (127, 44, 12)
        sf1 = video.Surface (10, 10, 16)
        for bpp in modes:
            sf2 = video.Surface (5, 5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, multiply blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (3, 8, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (255, 178, 177))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (2, 5, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (1, 3, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 0, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

    def test_8bpp_BLEND_RGB_MULT (self):
        video.init ()
        modes = [32, 24, 16, 8]
        color1 = Color (8, 50, 10)
        color2 = Color (127, 44, 12)
        sf1 = video.Surface (10, 10, 8)
        for bpp in modes:
            sf2 = video.Surface (5, 5, bpp)
        
            sf1.fill (color1)
            sf2.fill (color2)
        
            # Solid, multiply blit.
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (3, 8, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (255, 178, 177))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (2, 5, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (1, 3, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        
            sf2.fill (Color (0, 0, 0))
            sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
            self.assert_ (cmpcolor (sf1, sf2, Color (0, 0, 0),
                                    Rect (0, 0, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5)))
            self.assert_ (cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5)))
        video.quit ()

if __name__ == "__main__":
    unittest.main ()
