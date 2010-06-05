import unittest
import pygame2
from pygame2.colorpalettes import CGAPALETTE
from pygame2 import Color, Rect
import pygame2.sdl.video as video
import pygame2.sdl.constants as constants

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
    def _cmpcolor (self, surface, source, color, area=None):
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
        failmsg = "%s != %s at (%d, %d) for bpp: (%d, %d)"
        for x in range (sx, sx + w):
            for y in range (sy, sy + h):
                cc = getat (x, y)
                self.failUnlessEqual \
                    (cc, c, failmsg % (cc, c, x, y, sfbpp, srcbpp))    

    def setUp (self):
        video.init ()

    def tearDown (self):
        video.quit ()

    def test_simple_32bpp_blit (self):
        # Simple 32bpp blit
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 32)
        for bpp in modes:
            sf2 = video.Surface (5,  5, bpp)
            if bpp == 8:
                sf2.set_palette (CGAPALETTE)

            sf1.fill (color1)
            sf2.fill (color2)
            c2 = sf2.get_at (0, 0)
            
            # if it is a 16bpp -> 32bpp blit, we have Color(0, 124, 0) as 
            # result (in SDL).
            if bpp == 16:
                c2 = Color (0, 124, 0)

            # Solid, destructive blit.
            sf1.blit (sf2)
            self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
            self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
            self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
            self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
    
    def test_simple_24bpp_blit (self):
        # Simple 24bpp blit
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 24)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
            if bpp == 8:
                sf2.set_palette (CGAPALETTE)
        
            sf1.fill (color1)
            sf2.fill (color2)
            c2 = sf2.get_at (0, 0)
            
            # if it is a 16bpp -> 32bpp blit, we have Color(0, 124, 0) as 
            # result (in SDL).
            if bpp == 16:
                c2 = Color (0, 124, 0)

            # Solid, destructive blit.
            sf1.blit (sf2)

            self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
            self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
            self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
            self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
    
    def test_simple_16bpp_blit (self):
        # Simple 16bpp blit
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 16)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, bpp)
            if bpp == 8:
                sf2.set_palette (CGAPALETTE)
        
            sf1.fill (color1)
            sf2.fill (color2)
            c2 = sf2.get_at (0, 0)
            # Solid, destructive blit.
            sf1.blit (sf2)

            self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
            self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
            self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
            self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
    
    def test_simple_8bpp_blit (self):
        # Simple 8bpp blit
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        sf1 = video.Surface (10, 10, 8)
        sf1.set_palette (CGAPALETTE)
        for bpp in modes:
            sf2 = video.Surface ( 5,  5, 8)
            if bpp == 8:
                sf2.set_palette (CGAPALETTE)
        
            sf1.fill (color1)
            sf2.fill (color2)
            c2 = sf2.get_at (0, 0)
            # Solid, destructive blit.
            sf1.blit (sf2)

            self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
            self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
            self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
            self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))

    def test_BLEND_RGB_ADD (self):
        modes = [32, 24, 16, 8]
        color1 = Color (127, 0, 0)
        color2 = Color (0, 127, 0)
        for bpp in modes:
            sf1 = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf1.set_palette (CGAPALETTE)
            for bpp2 in modes:
                sf2 = video.Surface ( 5,  5, bpp2)
                if bpp2 == 8:
                    sf2.set_palette (CGAPALETTE)
        
                sf1.fill (color1)
                sf2.fill (color2)
                c2 = sf1.get_at (0, 0) + sf2.get_at (0, 0)
        
                # Solid, additive blit.
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                sf2.fill (Color (50, 127, 0))
                # Color (177, 254, 0)
                c2 = sf1.get_at (0, 0) + sf2.get_at (0, 0)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
                
                # Color (227, 255, 0)
                c2 = sf1.get_at (0, 0) + sf2.get_at (0, 0)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                sf2.fill (Color (0, 0, 0))
                # Color (227, 254, 0)
                c2 = sf1.get_at (0, 0) + sf2.get_at (0, 0)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_ADD)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))

    def test_BLEND_RGB_SUB (self):
        modes = [32, 24, 16, 8]
        color1 = Color (255, 255, 255)
        color2 = Color (0, 127, 0)
        for bpp in modes:
            sf1 = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf1.set_palette (CGAPALETTE)
            for bpp2 in modes:
                sf2 = video.Surface ( 5,  5, bpp2)
                if bpp2 == 8:
                    sf2.set_palette (CGAPALETTE)
        
                sf1.fill (color1)
                sf2.fill (color2)
                c2 = sf1.get_at (0, 0) - sf2.get_at (0, 0)
        
                # Solid, subtractive blit.
                
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
                # Color (255, 128, 255)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
                
                sf2.fill (Color (128, 20, 0))
                # Color (127, 108, 255)
                c2 = sf1.get_at (0, 0) - sf2.get_at (0, 0)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
                
                # Color (0, 88, 255)
                c2 = sf1.get_at (0, 0) - sf2.get_at (0, 0)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
                
                sf2.fill (Color (0, 0, 0))
                # Color (0, 88, 255)
                c2 = sf1.get_at (0, 0) - sf2.get_at (0, 0)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_SUB)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))

    def test_BLEND_RGB_MAX (self):
        modes = [32, 24, 16, 8]
        color1 = Color (30, 10, 100)
        color2 = Color (0, 127, 24)
        for bpp in modes:
            sf1 = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf1.set_palette (CGAPALETTE)
            for bpp2 in modes:
                sf2 = video.Surface ( 5,  5, bpp2)
                if bpp2 == 8:
                    sf2.set_palette (CGAPALETTE)
        
                sf1.fill (color1)
                sf2.fill (color2)
            
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (max(ca.r, cb.r), max(ca.g, cb.g), max(ca.b, cb.b))
                
                # Solid, maximum blit.
                
                # Color (30, 127, 100)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
            
                # Color (128, 127, 144)
                sf2.fill (Color (128, 20, 144))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (max(ca.r, cb.r), max(ca.g, cb.g), max(ca.b, cb.b))
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
            
                # Color (128, 127, 144)
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (max(ca.r, cb.r), max(ca.g, cb.g), max(ca.b, cb.b))
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                # Color (128, 127, 144)
                sf2.fill (Color (0, 0, 0))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (max(ca.r, cb.r), max(ca.g, cb.g), max(ca.b, cb.b))
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_MAX)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))

    def test_BLEND_RGB_MIN (self):
        modes = [32, 24, 16, 8]
        color1 = Color (255, 100, 12)
        color2 = Color (133, 127, 16)
        for bpp in modes:
            sf1 = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf1.set_palette (CGAPALETTE)
            for bpp2 in modes:
                sf2 = video.Surface ( 5,  5, bpp2)
                if bpp2 == 8:
                    sf2.set_palette (CGAPALETTE)
        
                sf1.fill (color1)
                sf2.fill (color2)
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (min(ca.r, cb.r), min(ca.g, cb.g), min(ca.b, cb.b))
            
                # Solid, minimum blit.
                
                # Color (133, 100, 12)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                # Color (128, 20, 12)
                sf2.fill (Color (128, 20, 144))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (min(ca.r, cb.r), min(ca.g, cb.g), min(ca.b, cb.b))
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
            
                # Color (128, 20, 12)
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (min(ca.r, cb.r), min(ca.g, cb.g), min(ca.b, cb.b))
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                # Color (0, 0, 0)
                sf2.fill (Color (0, 0, 0))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (min(ca.r, cb.r), min(ca.g, cb.g), min(ca.b, cb.b))
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_MIN)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))

    def test_BLEND_RGB_MULT (self):
        modes = [32, 24, 16, 8]
        color1 = Color (8, 50, 10)
        color2 = Color (127, 44, 12)
        for bpp in modes:
            sf1 = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf1.set_palette (CGAPALETTE)
            for bpp2 in modes:
                sf2 = video.Surface (5, 5, bpp2)
                if bpp2 == 8:
                    sf2.set_palette (CGAPALETTE)
        
                sf1.fill (color1)
                sf2.fill (color2)
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color ((ca.r * cb.r) >> 8, (ca.g * cb.g) >> 8,
                            (ca.b * cb.b) >> 8)
            
                # Solid, multiply blit.
                
                # Color (3, 8, 0)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                # Color (2, 5, 0)
                sf2.fill (Color (255, 178, 177))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color ((ca.r * cb.r) >> 8, (ca.g * cb.g) >> 8,
                            (ca.b * cb.b) >> 8)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
            
                # Color (1, 3, 0)
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color ((ca.r * cb.r) >> 8, (ca.g * cb.g) >> 8,
                            (ca.b * cb.b) >> 8)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                # Color (0, 0, 0)
                sf2.fill (Color (0, 0, 0))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color ((ca.r * cb.r) >> 8, (ca.g * cb.g) >> 8,
                            (ca.b * cb.b) >> 8)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_MULT)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))

    def test_BLEND_RGB_OR (self):
        modes = [32, 24, 16, 8]
        color1 = Color (8, 50, 10)
        color2 = Color (127, 44, 12)
        for bpp in modes:
            sf1 = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf1.set_palette (CGAPALETTE)
            for bpp2 in modes:
                sf2 = video.Surface (5, 5, bpp2)
                if bpp2 == 8:
                    sf2.set_palette (CGAPALETTE)

                sf1.fill (color1)
                sf2.fill (color2)
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (ca.r | cb.r, ca.g | cb.g, ca.b | cb.b)

                # Solid OR blit.
                
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_OR)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                sf2.fill (Color (255, 178, 177))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (ca.r | cb.r, ca.g | cb.g, ca.b | cb.b)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_OR)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
            
                sf2.fill (Color (0, 0, 0))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (ca.r | cb.r, ca.g | cb.g, ca.b | cb.b)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_OR)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))

    def test_BLEND_RGB_AND (self):
        modes = [32, 24, 16, 8]
        color1 = Color (8, 50, 10)
        color2 = Color (127, 44, 12)
        for bpp in modes:
            sf1 = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf1.set_palette (CGAPALETTE)
            for bpp2 in modes:
                sf2 = video.Surface (5, 5, bpp2)
                if bpp2 == 8:
                    sf2.set_palette (CGAPALETTE)

                sf1.fill (color1)
                sf2.fill (color2)
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (ca.r & cb.r, ca.g & cb.g, ca.b & cb.b)

                # Solid, AND blit.
                
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_AND)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                sf2.fill (Color (255, 178, 177))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (ca.r & cb.r, ca.g & cb.g, ca.b & cb.b)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_AND)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
            
                sf2.fill (Color (0, 0, 0))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (ca.r & cb.r, ca.g & cb.g, ca.b & cb.b)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_AND)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))

    def test_BLEND_RGB_XOR (self):
        modes = [32, 24, 16, 8]
        color1 = Color (8, 50, 10)
        color2 = Color (127, 44, 12)
        for bpp in modes:
            sf1 = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf1.set_palette (CGAPALETTE)
            for bpp2 in modes:
                sf2 = video.Surface (5, 5, bpp2)
                if bpp2 == 8:
                    sf2.set_palette (CGAPALETTE)

                sf1.fill (color1)
                sf2.fill (color2)
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (ca.r ^ cb.r, ca.g ^ cb.g, ca.b ^ cb.b)

                # Solid XOR blit.
                
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_XOR)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                sf2.fill (Color (255, 178, 177))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (ca.r ^ cb.r, ca.g ^ cb.g, ca.b ^ cb.b)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_XOR)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
            
                sf2.fill (Color (0, 0, 0))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (ca.r ^ cb.r, ca.g ^ cb.g, ca.b ^ cb.b)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_XOR)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))

    def test_BLEND_RGB_DIFF (self):
        modes = [32, 24, 16, 8]
        color1 = Color (8, 50, 10)
        color2 = Color (127, 44, 12)
        for bpp in modes:
            sf1 = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf1.set_palette (CGAPALETTE)
            for bpp2 in modes:
                sf2 = video.Surface (5, 5, bpp2)
                if bpp2 == 8:
                    sf2.set_palette (CGAPALETTE)

                sf1.fill (color1)
                sf2.fill (color2)
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (abs(cb.r - ca.r), abs(cb.g - ca.g),
                            abs(cb.b - ca.b))

                # Solid DIFF blit.
                
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_DIFF)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                sf2.fill (Color (255, 178, 177))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (abs(cb.r - ca.r), abs(cb.g - ca.g),
                            abs(cb.b - ca.b))
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_DIFF)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
            
                sf2.fill (Color (0, 0, 0))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (abs(cb.r - ca.r), abs(cb.g - ca.g),
                            abs(cb.b - ca.b))
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_DIFF)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))

    def test_BLEND_RGB_AVG (self):
        modes = [32, 24, 16, 8]
        color1 = Color (8, 50, 10)
        color2 = Color (127, 44, 12)
        for bpp in modes:
            sf1 = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf1.set_palette (CGAPALETTE)
            for bpp2 in modes:
                sf2 = video.Surface (5, 5, bpp2)
                if bpp2 == 8:
                    sf2.set_palette (CGAPALETTE)

                sf1.fill (color1)
                sf2.fill (color2)
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color ((ca.r + cb.r) >> 1, (ca.g + cb.g) >> 1,
                            (ca.b + cb.b) >> 1)

                # Solid AVG blit.
                
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_AVG)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                sf2.fill (Color (255, 178, 177))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color ((ca.r + cb.r) >> 1, (cb.g + ca.g) >> 1,
                            (ca.b + cb.b) >> 1)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_AVG)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
            
                sf2.fill (Color (0, 0, 0))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color ((ca.r + cb.r) >> 1, (ca.g + cb.g) >> 1,
                            (ca.b + cb.b) >> 1)
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_AVG)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))

    def test_BLEND_RGB_SCREEN (self):
        modes = [32, 24, 16, 8]
        color1 = Color (8, 50, 10)
        color2 = Color (127, 44, 12)
        for bpp in modes:
            sf1 = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf1.set_palette (CGAPALETTE)
            for bpp2 in modes:
                sf2 = video.Surface (5, 5, bpp2)
                if bpp2 == 8:
                    sf2.set_palette (CGAPALETTE)

                sf1.fill (color1)
                sf2.fill (color2)
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (255 - ((255 - ca.r) * (255 - cb.r) >> 8),
                            255 - ((255 - ca.g) * (255 - cb.g) >> 8),
                            255 - ((255 - ca.b) * (255 - cb.b) >> 8))

                # Solid SCREEN blit.
                
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_SCREEN)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
        
                sf2.fill (Color (255, 178, 177))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (255 - ((255 - ca.r) * (255 - cb.r) >> 8),
                            255 - ((255 - ca.g) * (255 - cb.g) >> 8),
                            255 - ((255 - ca.b) * (255 - cb.b) >> 8))
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_SCREEN)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))
            
                sf2.fill (Color (0, 0, 0))
                ca, cb = sf1.get_at (0, 0), sf2.get_at (0, 0)
                c2 = Color (255 - ((255 - ca.r) * (255 - cb.r) >> 8),
                            255 - ((255 - ca.g) * (255 - cb.g) >> 8),
                            255 - ((255 - ca.b) * (255 - cb.b) >> 8))
                sf1.blit (sf2, blendargs=constants.BLEND_RGB_SCREEN)
                self._cmpcolor (sf1, sf2, c2, Rect (0, 0, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (0, 5, 5, 5))
                self._cmpcolor (sf1, sf2, color1, Rect (5, 0, 5, 5))

if __name__ == "__main__":
    unittest.main ()
