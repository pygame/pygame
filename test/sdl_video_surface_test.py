import sys
try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

try:
    import StringIO as stringio
except ImportError:
    import io as stringio

import pygame2
import pygame2.sdl.video as video
import pygame2.sdl.image as image
import pygame2.sdlimage as sdlimage
import pygame2.sdl.constants as constants

def cmppixels (sf1, sf2):
    # Simple pixel comparision
    w1, h1 = sf1.size
    w2, h2 = sf2.size
    w, h = min (w1, w2), min (h1, h2)
    getat1 = sf1.get_at
    getat2 = sf2.get_at
    for x in range (w):
        for y in range (h):
            if getat1 (x, y) != getat2 (x, y):
                return False
    return True

def cmpcolor (sf, color, area=None):
    # Simple color comparision with clip area support
    getat = sf.get_at
    sx, sy = 0, 0
    w, h = sf.size
    if area:
        sx, sy = area.x, area.y 
        w, h = area.w, area.h
    for x in range (sx, sx + w):
        for y in range (sy, sy + h):
            if getat (x, y) != color:
                return False
    return True

class SDLVideoSurfaceTest (unittest.TestCase):

    def todo_test_pygame2_sdl_video_Surface_blit(self):
        # This is done in sdl_video_surface_blit_test.py
        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_clip_rect(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.clip_rect:

        # Gets or sets the current clipping rectangle for
        # operations on the Surface.
        video.init ()
        video.quit ()
        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_convert(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.convert:

        # convert ([pixelformat, flags]) -> Surface
        # 
        # Converts the Surface to the desired pixel format.
        # 
        # Converts the Surface to the desired pixel format. If no format
        # is given, the active display format is used, which will be the
        # fastest for blit operations to the screen. The flags are the
        # same as for surface creation.
        # 
        # This creates a new, converted surface and leaves the original one
        # untouched.
        video.init ()
        video.quit ()
        self.fail() 

    def test_pygame2_sdl_video_Surface_copy(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.copy:

        # copy () -> Surface
        # 
        # Creates an exact copy of the Surface and its image data.
        video.init ()
        sf = video.Surface (10, 10, 32)
        sfcopy = sf.copy ()
        
        self.assert_ (sf.size == sfcopy.size)
        self.assert_ (sf.format.bits_per_pixel == sfcopy.format.bits_per_pixel)
        self.assert_ (sf.format.masks == sfcopy.format.masks)
        self.assert_ (cmppixels (sf, sfcopy) == True)
        
        sf.fill (pygame2.Color (200, 100, 0))
        sfcopy = sf.copy ()
        self.assert_ (cmppixels (sf, sfcopy) == True)
        video.quit ()

    def test_pygame2_sdl_video_Surface_fill(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.fill:

        # fill (color[, dstrect, blendargs]) -> None
        # 
        # Fills the Surface with a color.
        # 
        # Fills the Surface with the desired color. The color does not need to
        # match the Surface's format, it will be converted implicitly to the
        # nearest appropriate color for its format.
        # 
        # The optional destination rectangle limits the color fill to
        # the specified area. The blendargs are the same as for the blit
        # operation, but compare the color with the specific Surface
        # pixel value.
        video.init ()
        sf = video.Surface (10, 20, 32)

        self.assert_ (cmpcolor (sf, pygame2.Color ("black")) == True)
        sf.fill (pygame2.Color ("cyan"))
        self.assert_ (cmpcolor (sf, pygame2.Color ("cyan")) == True)
        
        sf.fill (pygame2.Color ("cyan"))
        self.assert_ (cmpcolor (sf, pygame2.Color ("cyan")) == True)

        sf.fill (pygame2.Color ("yellow"), pygame2.Rect (5, 5, 4, 5))
        self.assert_ (cmpcolor (sf, pygame2.Color ("yellow"),
                      pygame2.Rect (5, 5, 4, 5)) == True)
        video.quit ()

    def test_pygame2_sdl_video_Surface_flags(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.flags:

        # The currently set flags for the Surface.
        video.init ()
        
        c = constants
        
        sf = video.Surface (10, 10, flags=c.SWSURFACE)
        self.assert_ ((sf.flags & c.SWSURFACE) == c.SWSURFACE)

        # HWSURFACE solely depends on the underlying hardware, additional
        # flags and whatever else, so we won't test for it here.
        
        # SDL does not set SRCCOLORKEY instantly on creation. Instead it is
        # set on applying a color key.
        sf = video.Surface (10, 10, flags=c.SRCCOLORKEY)
        self.assert_ ((sf.flags & c.SRCCOLORKEY) == 0)
        
        sf = video.Surface (10, 10, flags=c.SRCALPHA)
        self.assert_ ((sf.flags & c.SRCALPHA) == c.SRCALPHA)

        sf = video.Surface (10, 10, flags=c.SWSURFACE|c.SRCALPHA)
        self.assert_ ((sf.flags & c.SWSURFACE) == c.SWSURFACE)
        self.assert_ ((sf.flags & c.SRCALPHA) == c.SRCALPHA)

        sf = video.Surface (10, 10, flags=c.SRCCOLORKEY|c.SRCALPHA)
        self.assert_ ((sf.flags & c.SRCCOLORKEY) == 0)
        self.assert_ ((sf.flags & c.SRCALPHA) == c.SRCALPHA)
        
        sf = video.Surface (10, 10, flags=c.SWSURFACE|c.SRCCOLORKEY|c.SRCALPHA)
        self.assert_ ((sf.flags & c.SRCCOLORKEY) == 0)
        self.assert_ ((sf.flags & c.SWSURFACE) == c.SWSURFACE)
        self.assert_ ((sf.flags & c.SRCALPHA) == c.SRCALPHA)

        video.quit ()

    def test_pygame2_sdl_video_Surface_flip(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.flip:

        # flip () -> None
        # 
        # Swaps the screen buffers for the Surface.
        # 
        # Swaps screen buffers for the Surface, causing a full update
        # and redraw of its whole area.
        video.init ()
        sf = video.Surface (10, 10)
        self.assert_ (sf.flip () == None)
        video.quit ()

    def todo_test_pygame2_sdl_video_Surface_format(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.format:

        # Gets the (read-only) pygame2.sdl.video.PixelFormat for this
        # Surface.
        video.init ()
        video.quit ()
        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_get_alpha(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.get_alpha:

        # get_alpha () -> int
        # 
        # Gets the current overall alpha value of the Surface.
        # 
        # Gets the current overall alpha value of the Surface. In case the
        # surface does not support alpha transparency (SRCALPHA flag not set),
        # None will be returned.
        video.init ()
        video.quit ()
        self.fail() 

    def test_pygame2_sdl_video_Surface_get_at(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.get_at:

        # get_at (x, y) -> Color
        # get_at (point) -> Color
        # 
        # Gets the Surface pixel value at the specified point.
        video.init ()
        sf = video.Surface (10, 10)

        color = pygame2.Color (100, 50, 20)
        sf.fill (color)
        color = sf.format.get_rgba (color)
        rect = pygame2.Rect (0, 0, 3, 7)
        for x in range (10):
            for y in range (10):
                rect.topleft = x, y
                self.assert_ (sf.get_at (x, y) == color)
                self.assert_ (sf.get_at ((x, y)) == color)
                self.assert_ (sf.get_at (rect) == color)
        video.quit ()

    def todo_test_pygame2_sdl_video_Surface_get_colorkey(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.get_colorkey:

        # get_colorkey () -> Color
        # 
        # Gets the colorkey for the Surface.
        # 
        # Gets the colorkey for the Surface or None in case it has no colorkey
        # (SRCCOLORKEY flag not set).
        video.init ()
        video.quit ()
        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_get_palette(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.get_palette:

        # get_palette () -> (Color, Color, ...)
        video.init ()
        video.quit ()
        self.fail() 

    def test_pygame2_sdl_video_Surface_h(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.h:

        # Gets the height of the Surface.
        video.init ()
        self.assert_ (video.Surface (10, 10).h == 10)
        self.assert_ (video.Surface (10, 1).h == 1)
        self.assert_ (video.Surface (10, 100).h == 100)
        self.assert_ (video.Surface (0, 0).h == 0)
        self.assert_ (video.Surface (0, 10).h == 10)
        self.assert_ (video.Surface (10, 0).h == 0)
        self.assert_ (video.Surface (2, 65535).h == 65535)
        self.assertRaises (ValueError, video.Surface, -10, 10)
        self.assertRaises (pygame2.Error, video.Surface, 68000, 10)
        video.quit ()

    def test_pygame2_sdl_video_Surface_height(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.height:

        # Gets the height of the Surface.
        video.init ()
        self.assert_ (video.Surface (10, 10).height == 10)
        self.assert_ (video.Surface (10, 1).height == 1)
        self.assert_ (video.Surface (10, 100).height == 100)
        self.assert_ (video.Surface (0, 0).height == 0)
        self.assert_ (video.Surface (0, 10).height == 10)
        self.assert_ (video.Surface (10, 0).height == 0)
        self.assert_ (video.Surface (2, 65535).height == 65535)
        self.assertRaises (ValueError, video.Surface, -10, 10)
        self.assertRaises (pygame2.Error, video.Surface, 68000, 10)
        video.quit ()

    def test_pygame2_sdl_video_Surface_lock(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.lock:

        # lock () -> None
        # 
        # Locks the Surface for a direct access to its internal pixel data.
        video.init ()
        sf = video.Surface (10, 10)
        self.assert_ (sf.lock () == None)
        self.assert_ (sf.unlock () == None)
        video.quit ()

    def test_pygame2_sdl_video_Surface_locked(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.locked:

        # Gets, whether the Surface is currently locked.
        video.init ()
        sf = video.Surface (10, 10)
        sf.lock ()
        self.assert_ (sf.locked == True)
        sf.unlock ()
        self.assert_ (sf.locked == False)

        for i in range (4):
            sf.lock () 
            self.assert_ (sf.locked == True)
        self.assert_ (sf.locked == True)
        
        for i in range (3):
            sf.unlock ()
            self.assert_ (sf.locked == True)
        sf.unlock ()
        self.assert_ (sf.locked == False)
        sf.unlock ()
        self.assert_ (sf.locked == False)
        video.quit ()

    def todo_test_pygame2_sdl_video_Surface_pitch(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.pitch:

        # Get the length of a surface scanline in bytes.
        video.init ()
        video.quit ()
        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_pixels(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.pixels:

        # Gets the pixel buffer of the Surface.
        video.init ()
        video.quit ()
        self.fail() 

    def test_pygame2_sdl_video_Surface_save(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.save:

        # save (file[, type]) -> None
        # 
        # Saves the Surface to a file.
        # 
        # Saves the Surface to a file. The file argument can be either a
        # file name or a file-like object to save the Surface to. The
        # optional type argument is required, if the file type cannot be
        # determined by the suffix.
        # 
        # Currently supported file types (suitable to pass as string for
        # the type argument) are:
        # 
        # * BMP
        # * TGA
        # * PNG
        # * JPEG (JPG)
        # 
        # If no type information is supplied and the file type cannot be
        # determined either, it will use TGA.
        video.init ()
        sf1 = video.Surface (16, 16, 32)
        sf1.fill (pygame2.Color ("red"))
        bufcreat = None
        if sys.version_info[0] >= 3:
            bufcreat = stringio.BytesIO
        else:
            bufcreat = stringio.StringIO
        buf = bufcreat ()
        
        sf1.save (buf, "bmp")
        buf.seek (0)
        sf2 = image.load_bmp (buf)
        self.assert_ (sf1.size == sf2.size)
        self.assert_ (cmppixels (sf1, sf2) == True)
        
        buf.seek (0)
        sf2 = sdlimage.load (buf, "bmp")
        self.assert_ (sf1.size == sf2.size)
        self.assert_ (cmppixels (sf1, sf2) == True)

        buf = bufcreat ()
        sf1.save (buf, "jpg")
        buf.seek (0)
        sf2 = sdlimage.load (buf, "jpg")
        self.assert_ (sf1.size == sf2.size)

        buf = bufcreat ()
        sf1.save (buf, "png")
        buf.seek (0)
        sf2 = sdlimage.load (buf, "png")
        self.assert_ (sf1.size == sf2.size)
        self.assert_ (cmppixels (sf1, sf2) == True)

        buf = bufcreat ()
        sf1.save (buf, "tga")
        buf.seek (0)
        sf2 = sdlimage.load (buf, "tga")
        self.assert_ (sf1.size == sf2.size)
        self.assert_ (cmppixels (sf1, sf2) == True)
        video.quit ()

    def todo_test_pygame2_sdl_video_Surface_set_alpha(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.set_alpha:

        # set_alpha (alpha[, flags]) -> None
        # 
        # Adjusts the alpha properties of the Surface.
        # 
        # TODO
        video.init ()
        video.quit ()
        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_set_at(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.set_at:

        # set_at (x, y, color) -> None
        # set_at (point, color) -> None
        # 
        # Sets the Surface pixel value at the specified point.
        video.init ()
        video.quit ()
        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_set_colorkey(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.set_colorkey:

        # set_colorkey (colorkey[, flags]) -> None
        # 
        # Adjusts the colorkey of the Surface.
        # 
        # TODO
        video.quit ()
        video.init ()
        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_set_colors(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.set_colors:

        # set_colors ((color1, color2, ...)[, first]) -> bool
        # 
        # Sets a portion of the colormap palette for the 8-bit Surface.
        # 
        # Sets a portion of the colormap palette for the 8-bit Surface,
        # starting at the desired first position. If the first position
        # plus length of the passed colormap exceeds the Surface palette
        # size, the palette will be unchanged and False returned.
        # 
        # If any other error occurs, False will be returned and the
        # Surface palette should be inspected for any changes.
        video.init ()
        video.quit ()
        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_set_palette(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.set_palette:

        # set_palette ((color1, color2, ...), flags[, first]) -> bool
        # 
        # Sets a portion of the palette for the given 8-bit surface.
        # 
        # Sets a portion of the color palette for the 8-bit Surface,
        # starting at the desired first position. If the first position
        # plus length of the passed colormap exceeds the Surface palette
        # size, the palette will be unchanged and False returned.
        # 
        # If any other error occurs, False will be returned and the
        # Surface palette should be inspected for any changes.
        video.init ()
        video.quit ()
        self.fail() 

    def test_pygame2_sdl_video_Surface_size(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.size:

        # Gets the size of the Surface.
        video.init ()
        self.assert_ (video.Surface (10, 10).size == (10, 10))
        self.assert_ (video.Surface (1, 10).size == (1, 10))
        self.assert_ (video.Surface (100, 10).size == (100, 10))
        self.assert_ (video.Surface (0, 0).size == (0, 0))
        self.assert_ (video.Surface (0, 10).size == (0, 10))
        self.assert_ (video.Surface (16383, 2).size == (16383, 2))
        self.assert_ (video.Surface (2, 65535).size == (2, 65535))
        self.assertRaises (ValueError, video.Surface, -10, 10)
        self.assertRaises (pygame2.Error, video.Surface, 68000, 10)
        video.quit ()

    def test_pygame2_sdl_video_Surface_unlock(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.unlock:

        # unlock () -> None
        # 
        # Unlocks the Surface, releasing the direct access to the pixel data.
        video.init ()
        sf = video.Surface (10, 10)
        self.assert_ (sf.lock () == None)
        self.assert_ (sf.unlock () == None)
        video.quit ()

    def todo_test_pygame2_sdl_video_Surface_update(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.update:

        # update ([rect]) -> None
        # update ([(rect1, rect2, ...)]) -> None
        # 
        # Updates the given area on the Surface.
        # 
        # Upates the given area (or areas, if a list of rects is passed)
        # on the Surface.
        video.init ()
        video.quit ()
        self.fail() 

    def test_pygame2_sdl_video_Surface_w(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.w:

        # Gets the width of the Surface.
        video.init ()
        self.assert_ (video.Surface (10, 10).w == 10)
        self.assert_ (video.Surface (1, 10).w == 1)
        self.assert_ (video.Surface (100, 10).w == 100)
        self.assert_ (video.Surface (0, 0).w == 0)
        self.assert_ (video.Surface (0, 10).w == 0)
        self.assert_ (video.Surface (10, 0).w == 10)
        self.assert_ (video.Surface (16383, 2).w == 16383)
        self.assertRaises (ValueError, video.Surface, -10, 10)
        self.assertRaises (pygame2.Error, video.Surface, 68000, 10)
        video.quit ()

    def test_pygame2_sdl_video_Surface_width(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.width:

        # Gets the width of the Surface.
        video.init ()
        self.assert_ (video.Surface (10, 10).width == 10)
        self.assert_ (video.Surface (1, 10).width == 1)
        self.assert_ (video.Surface (100, 10).width == 100)
        self.assert_ (video.Surface (0, 0).width == 0)
        self.assert_ (video.Surface (0, 10).width == 0)
        self.assert_ (video.Surface (10, 0).width == 10)
        self.assert_ (video.Surface (16383, 2).width == 16383)
        self.assertRaises (ValueError, video.Surface, -10, 10)
        self.assertRaises (pygame2.Error, video.Surface, 68000, 10)
        video.quit ()

if __name__ == "__main__":
    unittest.main ()
