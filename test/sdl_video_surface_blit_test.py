import sys
try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import pygame2
from pygame2 import Color, Rect
import pygame2.sdl.video as video
import pygame2.sdl.constants as constants

def cmpcolor (surface, color, area=None):
    # Simple color comparision with clip area support
    getat = surface.get_at
    sx, sy = 0, 0
    w, h = surface.size
    if area:
        sx, sy = area.x, area.y 
        w, h = area.w, area.h

    c = surface.format.map_rgba (color)
    for x in range (sx, sx + w):
        for y in range (sy, sy + h):
            if getat (x, y) != c:
                #print (x, y, getat (x, y), c, color)
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
        sf1 = video.Surface (10, 10, 32)
        sf2 = video.Surface ( 5,  5, 32)
        
        sf1.fill (Color (127, 0, 0))
        sf2.fill (Color (0, 127, 0))
        
        # Solid, destructive blit.
        sf1.blit (sf2)
        self.assert_ (cmpcolor (sf1, Color (0, 127, 0), Rect (0, 0, 5, 5)))
        self.assert_ (cmpcolor (sf1, Color (127, 0, 0), Rect (5, 5, 5, 5)))
        self.assert_ (cmpcolor (sf1, Color (127, 0, 0), Rect (0, 5, 5, 5)))
        self.assert_ (cmpcolor (sf1, Color (127, 0, 0), Rect (5, 0, 5, 5)))
        video.quit ()
    
    def test_simple_24bpp_blit (self):
        # Simple 24bpp blit
        video.init ()
        sf1 = video.Surface (10, 10, 24)
        sf2 = video.Surface ( 5,  5, 24)
        
        sf1.fill (Color (127, 0, 0))
        sf2.fill (Color (0, 127, 0))
        
        # Solid, destructive blit.
        sf1.blit (sf2)
        self.assert_ (cmpcolor (sf1, Color (0, 127, 0), Rect (0, 0, 5, 5)))
        self.assert_ (cmpcolor (sf1, Color (127, 0, 0), Rect (5, 5, 5, 5)))
        self.assert_ (cmpcolor (sf1, Color (127, 0, 0), Rect (0, 5, 5, 5)))
        self.assert_ (cmpcolor (sf1, Color (127, 0, 0), Rect (5, 0, 5, 5)))
        video.quit ()
    
    def todo_test_simple_16bpp_blit (self):
        # Simple 16bpp blit
        video.init ()
        sf1 = video.Surface (10, 10, 16)
        sf2 = video.Surface ( 5,  5, 16)
        
        sf1.fill (Color (127, 0, 0))
        sf2.fill (Color (0, 127, 0))
        
        # Solid, destructive blit.
        sf1.blit (sf2)
        self.assert_ (cmpcolor (sf1, Color (0, 127, 0), Rect (0, 0, 5, 5)))
        self.assert_ (cmpcolor (sf1, Color (127, 0, 0), Rect (5, 5, 5, 5)))
        self.assert_ (cmpcolor (sf1, Color (127, 0, 0), Rect (0, 5, 5, 5)))
        self.assert_ (cmpcolor (sf1, Color (127, 0, 0), Rect (5, 0, 5, 5)))
        video.quit ()
    
    def test_simple_8bpp_blit (self):
        # Simple 8bpp blit
        video.init ()
        sf1 = video.Surface (10, 10, 8)
        sf2 = video.Surface ( 5,  5, 8)
        
        sf1.fill (Color (127, 0, 0))
        sf2.fill (Color (0, 127, 0))
        
        # Solid, destructive blit.
        sf1.blit (sf2)
        self.assert_ (cmpcolor (sf1, Color (0, 127, 0), Rect (0, 0, 5, 5)))
        self.assert_ (cmpcolor (sf1, Color (127, 0, 0), Rect (5, 5, 5, 5)))
        self.assert_ (cmpcolor (sf1, Color (127, 0, 0), Rect (0, 5, 5, 5)))
        self.assert_ (cmpcolor (sf1, Color (127, 0, 0), Rect (5, 0, 5, 5)))
        video.quit ()

if __name__ == "__main__":
    unittest.main ()
