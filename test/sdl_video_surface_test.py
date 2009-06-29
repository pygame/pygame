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
from pygame2.colorpalettes import CGAPALETTE
from pygame2 import Rect, Color
import pygame2.sdl.video as video
import pygame2.sdl.image as image
import pygame2.sdlimage as sdlimage
import pygame2.sdl.constants as constants

class SDLVideoSurfaceTest (unittest.TestCase):

    def _cmppixels (self, sf1, sf2):
        # Simple pixel comparision
        w1, h1 = sf1.size
        w2, h2 = sf2.size
        w, h = min (w1, w2), min (h1, h2)
        getat1 = sf1.get_at
        getat2 = sf2.get_at
        failmsg = "%s != %s at (%d, %d)"
        for x in range (w):
            for y in range (h):
                self.failUnlessEqual (getat1 (x, y), getat2 (x, y),
                    failmsg % (getat1 (x, y), getat2 (x, y), x, y))

    def _cmpcolor (self, sf, color, area=None):
        # Simple color comparision with clip area support
        getat = sf.get_at
        sx, sy = 0, 0
        w, h = sf.size
        if area:
            sx, sy = area.x, area.y 
            w, h = area.w, area.h
        c = sf.format.get_rgba (color)
        failmsg = "%s != %s at (%d, %d)"
        for x in range (sx, sx + w):
            for y in range (sy, sy + h):
                self.failUnlessEqual (getat (x, y), c,
                    failmsg % (getat (x, y), c, x, y))
    
    def test_pygame2_sdl_video_Surface_blit(self):
        # This is done in sdl_video_surface_blit_test.py
        pass

    def test_pygame2_sdl_video_Surface_clip_rect(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.clip_rect:

        # Gets or sets the current clipping rectangle for
        # operations on the Surface.
        video.init ()
        modes = [32, 24, 16, 8]
        for bpp in modes:
            sf = video.Surface (10, 10, bpp)
            sf.fill (Color (255, 255, 255))
            sf.clip_rect = Rect (3, 3, 3, 3)
            self.assertEqual (sf.clip_rect, Rect (3, 3, 3, 3))
            
            sf.fill (Color (255, 0, 0))
            self._cmpcolor (sf, Color (255, 255, 255), Rect (0, 0,  3, 10))
            self._cmpcolor (sf, Color (255, 255, 255), Rect (0, 0, 10,  2))
            self._cmpcolor (sf, Color (255, 255, 255), Rect (6, 0,  4, 10))
            self._cmpcolor (sf, Color (255, 255, 255), Rect (0, 6, 10,  4))
            self._cmpcolor (sf, Color (255, 0, 0), Rect (3, 3, 3, 3))
            
            sf.clip_rect = None
            sf.fill (Color (255, 255, 255))
            self._cmpcolor (sf, Color (255, 255, 255))
            
            sf.clip_rect = Rect (3, 3, 3, 3)
            sf.fill (Color (255, 0, 0))
            sf.fill (Color (0, 255, 0), Rect (4, 5, 10, 10))
            self._cmpcolor (sf, Color (255, 255, 255), Rect (0, 0,  3, 10))
            self._cmpcolor (sf, Color (255, 255, 255), Rect (0, 0, 10,  2))
            self._cmpcolor (sf, Color (255, 0, 0), Rect (3, 3, 3, 2))
            self._cmpcolor (sf, Color (255, 0, 0), Rect (3, 3, 1, 3))
            self._cmpcolor (sf, Color (0, 255, 0), Rect (4, 5, 2, 1))
            self._cmpcolor (sf, Color (255, 255, 255), Rect (6, 0,  4, 10))
            self._cmpcolor (sf, Color (255, 255, 255), Rect (0, 6, 10,  4))
        
        video.quit ()

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
        modes = [32, 24, 16, 8]
        for bpp in modes:
            sf = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf.set_palette (CGAPALETTE)
            sfcopy = sf.copy ()
        
            self.assertEqual (sf.size, sfcopy.size)
            self.assertEqual (sf.format.bits_per_pixel,
                              sfcopy.format.bits_per_pixel)
            self.assertEqual (sf.format.masks, sfcopy.format.masks)
            self._cmppixels (sf, sfcopy)
        
            sf.fill (Color (200, 100, 0))
            sfcopy = sf.copy ()
            self._cmppixels (sf, sfcopy)
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
        modes = [32, 24, 16, 8]
        for bpp in modes:
            sf = video.Surface (10, 20, bpp)
            if bpp == 8:
                sf.set_palette (CGAPALETTE)

            self._cmpcolor (sf, Color ("black"))
            sf.fill (Color ("cyan"))
            self._cmpcolor (sf, Color ("cyan"))

            sf.fill (Color ("cyan"))
            self._cmpcolor (sf, Color ("cyan"))

            sf.fill (Color ("yellow"), Rect (5, 5, 4, 5))
            self._cmpcolor (sf, Color ("yellow"), Rect (5, 5, 4, 5))
        video.quit ()

    def test_pygame2_sdl_video_Surface_flags(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.flags:

        # The currently set flags for the Surface.
        video.init ()
        # Add 24bpp and 8bpp alpha masks.
        modes = [32, 16]
        c = constants
        for bpp in modes:
        
            sf = video.Surface (10, 10, bpp, flags=c.SWSURFACE)
            self.assertEqual ((sf.flags & c.SWSURFACE), c.SWSURFACE)

            # HWSURFACE solely depends on the underlying hardware, additional
            # flags and whatever else, so we won't test for it here.
        
            # SDL does not set SRCCOLORKEY instantly on creation. Instead it is
            # set on applying a color key.
            sf = video.Surface (10, 10, bpp, flags=c.SRCCOLORKEY)
            self.assertEqual ((sf.flags & c.SRCCOLORKEY), 0)
        
            sf = video.Surface (10, 10, bpp, flags=c.SRCALPHA)
            self.assertEqual ((sf.flags & c.SRCALPHA), c.SRCALPHA)

            sf = video.Surface (10, 10, bpp, flags=c.SWSURFACE|c.SRCALPHA)
            self.assertEqual ((sf.flags & c.SWSURFACE), c.SWSURFACE)
            self.assertEqual ((sf.flags & c.SRCALPHA), c.SRCALPHA)

            sf = video.Surface (10, 10, bpp, flags=c.SRCCOLORKEY|c.SRCALPHA)
            self.assertEqual ((sf.flags & c.SRCCOLORKEY), 0)
            self.assertEqual ((sf.flags & c.SRCALPHA), c.SRCALPHA)

            sf = video.Surface (10, 10, bpp,
                                flags=c.SWSURFACE|c.SRCCOLORKEY|c.SRCALPHA)
            self.assertEqual ((sf.flags & c.SRCCOLORKEY), 0)
            self.assertEqual ((sf.flags & c.SWSURFACE), c.SWSURFACE)
            self.assertEqual ((sf.flags & c.SRCALPHA), c.SRCALPHA)

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
        modes =  [32, 24, 16, 8]
        for bpp in modes:
            sf = video.Surface (10, 10, bpp)
            self.assert_ (sf.flip () == None)
        video.quit ()

    def test_pygame2_sdl_video_Surface_format(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.format:

        # Gets the (read-only) pygame2.sdl.video.PixelFormat for this
        # Surface.
        video.init ()
        modes =  [32, 24, 16, 8]
        for bpp in modes:
            sf = video.Surface (10, 10, bpp)
            fmt = sf.format
            self.assertEqual (type (fmt), video.PixelFormat)
            self.assertEqual (fmt.bits_per_pixel, bpp)
            self.assertEqual (fmt.bytes_per_pixel, bpp // 8)
        video.quit ()

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
        modes = [32, 24, 16, 8]
        for bpp in modes:
            sf = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf.set_palette (CGAPALETTE)

            color = Color (100, 50, 20)
            sf.fill (color)
            color = sf.format.get_rgba (color)
            rect = Rect (0, 0, 3, 7)
            for x in range (10):
                for y in range (10):
                    rect.topleft = x, y
                    self.assertEqual (sf.get_at (x, y), color)
                    self.assertEqual (sf.get_at ((x, y)), color)
                    self.assertEqual (sf.get_at (rect), color)
        video.quit ()

    def todo_test_pygame2_sdl_video_Surface_get_colorkey(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.get_colorkey:

        # get_colorkey () -> Color
        # 
        # Gets the colorkey for the Surface.
        # 
        # Gets the colorkey for the Surface or None in case it has no colorkey
        # (SRCCOLORKEY flag not set).
        
        # This fails badly.
        video.init ()
        modes = [32, 24, 16, 8]
        for bpp in modes:
            sf = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf.set_palette (CGAPALETTE)
            self.assertEqual (sf.get_colorkey (), None)
            sf = video.Surface (10, 10, bpp, flags=constants.SRCCOLORKEY)
            if bpp == 8:
                sf.set_palette (CGAPALETTE)
            self.assertEqual (sf.get_colorkey (), None)
        
            sf = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf.set_palette (CGAPALETTE)
            color = Color (255, 0, 0)
            key = sf.format.get_rgba (color)
            self.assertTrue (sf.set_colorkey (color))
            self.assertEqual (sf.get_colorkey (), key)
        
            # TODO: something wicked happens here.
            # sf = video.Surface (10, 10, bpp, flags=constants.SRCCOLORKEY)
            # color = Color (255, 0, 0)
            # key = sf.format.get_rgba (color)
            # self.assertEqual (sf.set_colorkey (color), True)
            # self.assertEqual (sf.get_colorkey (), key)
        video.quit ()

    def todo_test_pygame2_sdl_video_Surface_get_palette(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.get_palette:

        # get_palette () -> (Color, Color, ...)
        #
        # Gets the palette colors used by the Surface.
        #
        # Gets the palette colors used by the Surface or None, if the Surface
        # does not use any palette.
        video.init ()
        video.quit ()
        self.fail() 

    def test_pygame2_sdl_video_Surface_h(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.h:

        # Gets the height of the Surface.
        video.init ()
        self.assertEqual (video.Surface (10, 10).h, 10)
        self.assertEqual (video.Surface (10, 1).h, 1)
        self.assertEqual (video.Surface (10, 100).h, 100)
        self.assertEqual (video.Surface (0, 0).h, 0)
        self.assertEqual (video.Surface (0, 10).h, 10)
        self.assertEqual (video.Surface (10, 0).h, 0)
        self.assertEqual (video.Surface (2, 65535).h, 65535)
        self.assertRaises (ValueError, video.Surface, -10, 10)
        self.assertRaises (pygame2.Error, video.Surface, 68000, 10)
        video.quit ()

    def test_pygame2_sdl_video_Surface_height(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.height:

        # Gets the height of the Surface.
        video.init ()
        self.assertEqual (video.Surface (10, 10).height, 10)
        self.assertEqual (video.Surface (10, 1).height, 1)
        self.assertEqual (video.Surface (10, 100).height, 100)
        self.assertEqual (video.Surface (0, 0).height, 0)
        self.assertEqual (video.Surface (0, 10).height, 10)
        self.assertEqual (video.Surface (10, 0).height, 0)
        self.assertEqual (video.Surface (2, 65535).height, 65535)
        self.assertRaises (ValueError, video.Surface, -10, 10)
        self.assertRaises (pygame2.Error, video.Surface, 68000, 10)
        video.quit ()

    def test_pygame2_sdl_video_Surface_lock(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.lock:

        # lock () -> None
        # 
        # Locks the Surface for a direct access to its internal pixel data.
        video.init ()
        modes = [32, 24, 16, 8]
        for bpp in modes:
            sf = video.Surface (10, 10, bpp)
            self.assert_ (sf.lock () == None)
            self.assert_ (sf.unlock () == None)
        video.quit ()

    def test_pygame2_sdl_video_Surface_locked(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.locked:

        # Gets, whether the Surface is currently locked.
        video.init ()
        modes = [32, 24, 16, 8]
        for bpp in modes:
            sf = video.Surface (10, 10, bpp)
            sf.lock ()
            self.assertTrue (sf.locked)
            sf.unlock ()
            self.assertFalse (sf.locked)

            for i in range (4):
                sf.lock () 
                self.assertTrue (sf.locked)
            self.assertTrue (sf.locked)
        
            for i in range (3):
                sf.unlock ()
                self.assertTrue (sf.locked)
            sf.unlock ()
            self.assertFalse (sf.locked)
            sf.unlock ()
            self.assertFalse (sf.locked)
        video.quit ()

    def test_pygame2_sdl_video_Surface_pitch(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.pitch:

        # Get the length of a surface scanline in bytes.
        video.init ()
        sf = video.Surface (10, 10, 32)
        self.assert_ (sf.pitch >= 40) # 10 * 4 bpp
        sf = video.Surface (10, 10, 24)
        self.assert_ (sf.pitch >= 30) # 10 * 3 bpp
        sf = video.Surface (10, 10, 16)
        self.assert_ (sf.pitch >= 20) # 10 * 2 bpp
        sf = video.Surface (10, 10, 8)
        self.assert_ (sf.pitch >= 10) # 10 * 1 bpp
        video.quit ()

    def test_pygame2_sdl_video_Surface_pixels(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.pixels:

        # Gets the pixel buffer of the Surface.
        video.init ()
        modes = [32, 24, 16, 8]
        for bpp in modes:
            sf = video.Surface (10, 10, bpp)
            self.assertFalse (sf.locked)
            self.assert_ (type (sf.pixels) == pygame2.BufferProxy)
            self.assertFalse (sf.locked)
            buf = sf.pixels
            self.assertTrue (sf.locked)
            self.assert_ (buf.length >= 10 * sf.format.bytes_per_pixel)
            del buf
            self.assertFalse (sf.locked)
        video.quit ()

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
        modes = [32, 24, 16, 8]
        for bpp in modes:
            sf1 = video.Surface (16, 16, bpp)
            if bpp == 8:
                sf1.set_palette (CGAPALETTE)
            sf1.fill (pygame2.Color ("red"))
            bufcreat = None
            if sys.version_info[0] >= 3:
                bufcreat = stringio.BytesIO
            else:
                bufcreat = stringio.StringIO
            buf = bufcreat ()
        
            sf1.save (buf, "bmp")
            buf.seek (0)
            sf2 = image.load_bmp (buf).convert (sf1.format)
            self.assertEqual (sf1.size, sf2.size)
            self._cmppixels (sf1, sf2)
        
            buf.seek (0)
            sf2 = sdlimage.load (buf, "bmp").convert (sf1.format)
            self.assertEqual (sf1.size, sf2.size)
            self._cmppixels (sf1, sf2)

            buf = bufcreat ()
            sf1.save (buf, "jpg")
            buf.seek (0)
            sf2 = sdlimage.load (buf, "jpg").convert (sf1.format)
            self.assertEqual (sf1.size, sf2.size)

            buf = bufcreat ()
            sf1.save (buf, "png")
            buf.seek (0)
            sf2 = sdlimage.load (buf, "png").convert (sf1.format)
            self.assertEqual (sf1.size, sf2.size)
            self._cmppixels (sf1, sf2)

            buf = bufcreat ()
            sf1.save (buf, "tga")
            buf.seek (0)
            sf2 = sdlimage.load (buf, "tga").convert (sf1.format)
            self.assertEqual (sf1.size, sf2.size)
            self._cmppixels (sf1, sf2)
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

    def test_pygame2_sdl_video_Surface_set_at(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.set_at:

        # set_at (x, y, color) -> None
        # set_at (point, color) -> None
        # 
        # Sets the Surface pixel value at the specified point.
        video.init ()
        modes = [32, 24, 16, 8]
        for bpp in modes:
            sf = video.Surface (10, 10, bpp)
            if bpp == 8:
                sf.set_palette (CGAPALETTE)

            color = Color (255, 255, 255)
            sf.fill (color)
            color = Color (100, 37, 44)
            cc = sf.format.get_rgba (color)
            rect = Rect (4, 4, 3, 6)
            for x in range (rect.x, rect.x + rect.w):
                for y in range (rect.y, rect.y + rect.h):
                    sf.set_at (x, y, color)
                    self.assertEqual (sf.get_at (x, y), cc)
                    sf.set_at ((x, y), color)
                    self.assertEqual (sf.get_at (x, y), cc)
            self._cmpcolor (sf, color, rect)
        video.quit ()

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
        self.assertEqual (video.Surface (10, 10).size, (10, 10))
        self.assertEqual (video.Surface (1, 10).size, (1, 10))
        self.assertEqual (video.Surface (100, 10).size, (100, 10))
        self.assertEqual (video.Surface (0, 0).size, (0, 0))
        self.assertEqual (video.Surface (0, 10).size, (0, 10))
        self.assertEqual (video.Surface (16383, 2).size, (16383, 2))
        self.assertEqual (video.Surface (2, 65535).size, (2, 65535))
        self.assertRaises (ValueError, video.Surface, -10, 10)
        self.assertRaises (pygame2.Error, video.Surface, 68000, 10)
        video.quit ()

    def test_pygame2_sdl_video_Surface_unlock(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.unlock:

        # unlock () -> None
        # 
        # Unlocks the Surface, releasing the direct access to the pixel data.
        video.init ()
        modes = [32, 24, 16, 8]
        for bpp in modes:
            sf = video.Surface (10, 10, bpp)
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
        self.assertEqual (video.Surface (10, 10).w, 10)
        self.assertEqual (video.Surface (1, 10).w, 1)
        self.assertEqual (video.Surface (100, 10).w, 100)
        self.assertEqual (video.Surface (0, 0).w, 0)
        self.assertEqual (video.Surface (0, 10).w, 0)
        self.assertEqual (video.Surface (10, 0).w, 10)
        self.assertEqual (video.Surface (16383, 2).w, 16383)
        self.assertRaises (ValueError, video.Surface, -10, 10)
        self.assertRaises (pygame2.Error, video.Surface, 68000, 10)
        video.quit ()

    def test_pygame2_sdl_video_Surface_width(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.width:

        # Gets the width of the Surface.
        video.init ()
        self.assertEqual (video.Surface (10, 10).width, 10)
        self.assertEqual (video.Surface (1, 10).width, 1)
        self.assertEqual (video.Surface (100, 10).width, 100)
        self.assertEqual (video.Surface (0, 0).width, 0)
        self.assertEqual (video.Surface (0, 10).width, 0)
        self.assertEqual (video.Surface (10, 0).width, 10)
        self.assertEqual (video.Surface (16383, 2).width, 16383)
        self.assertRaises (ValueError, video.Surface, -10, 10)
        self.assertRaises (pygame2.Error, video.Surface, 68000, 10)
        video.quit ()

    def test_pygame2_sdl_video_Surface_scroll (self):

        # __doc__ (as of 2009-06-24) for pygame2.sdl.video.Surface.width:

        # Scrolls the Surface in place.

        # Move the Surface contents by *dx* pixels right and *dy* pixels
        # down. dx and dy may be negative for left and up scrolls
        # respectively. Areas of the surface that are not overwritten
        # retain their original pixel values. Scrolling is contained by
        # the Surface clip area. It is safe to have *dx* and *dy* values
        # that exceed the surface size.
        video.init ()

        scrolls = [(8, 2, 3),
                   (16, 2, 3),
                   (24, 2, 3),
                   (32, 2, 3),
                   (32, -1, -3),
                   (32, 0, 0),
                   (32, 11, 0),
                   (32, 0, 11),
                   (32, -11, 0),
                   (32, 0, -11),
                   (32, -11, 2),
                   (32, 2, -11)]
        for bitsize, dx, dy in scrolls:
            if bitsize == 8:
                masks = (0xFF >> 6 << 5, 0xFF >> 5 << 2, 0xFF >> 6, 0)
                surf = video.Surface((10, 10), bitsize)
            else:
                surf = video.Surface((10, 10), bitsize)
            surf.fill(Color(255, 0, 0))
            surf.fill(Color(0, 255, 0), (2, 2, 2, 2,))
            comp = surf.copy()
            comp.blit(surf, (dx, dy))
            surf.scroll(dx, dy)
            w, h = surf.size
            for x in range(w):
                for y in range(h):
                    self.failUnlessEqual(surf.get_at((x, y)),
                                         comp.get_at((x, y)),
                                         "%s != %s, bpp:, %i, x: %i, y: %i" %
                                         (surf.get_at((x, y)),
                                          comp.get_at((x, y)),
                                          bitsize, dx, dy))
        # Confirm clip rect containment
        surf = video.Surface((20, 13), 32)
        surf.fill(Color(255, 0, 0))
        surf.fill(Color(0, 255, 0), (7, 1, 6, 6))
        comp = surf.copy()
        clip = Rect(3, 1, 8, 14)
        surf.clip_rect = clip
        comp.clip_rect = clip
        comp.blit(surf, (clip.x + 2, clip.y + 3), surf.clip_rect)
        surf.scroll(2, 3)
        w, h = surf.size
        for x in range(w):
            for y in range(h):
                self.failUnlessEqual(surf.get_at((x, y)),
                                     comp.get_at((x, y)))
        # Confirm keyword arguments and per-pixel alpha
        spot_color = Color(0, 255, 0, 128)
        surf = video.Surface((4, 4), 32, constants.SRCALPHA)
        surf.fill(Color(255, 0, 0, 255))
        surf.set_at((1, 1), spot_color)
        surf.scroll(dx=1)
        self.failUnlessEqual(surf.get_at((2, 1)), spot_color)
        surf.scroll(dy=1)
        self.failUnlessEqual(surf.get_at((2, 2)), spot_color)
        surf.scroll(dy=1, dx=1)
        self.failUnlessEqual(surf.get_at((3, 3)), spot_color)
        surf.scroll(dx=-3, dy=-3)
        self.failUnlessEqual(surf.get_at((0, 0)), spot_color)
        
        sf = video.Surface (20, 20)
        sf.fill (Color (255, 0, 0), Rect (10, 10, 20, 20))
        self._cmpcolor (sf, Color (255, 0, 0), Rect (10, 10, 10, 10))
        sf.scroll (0, -5)
        self._cmpcolor (sf, Color (255, 0, 0), Rect (10, 5, 10, 15))
        sf.scroll (-10, 0)
        self._cmpcolor (sf, Color (255, 0, 0), Rect (0, 5, 10, 15))
        self._cmpcolor (sf, Color (255, 0, 0), Rect (10, 10, 10, 10))

        sf.fill (Color (0))
        sf.fill (Color (255, 0, 0), Rect (0, 0, 5, 5))
        sf.scroll (-10, -10)
        self._cmpcolor (sf, Color (0, 0, 0))
        video.quit ()
        
    def test_pygame2_sdl_video_Surface___repr__(self):
        video.init ()
        sf = video.Surface (10, 10, 8)
        text = "<Surface 10x10@8bpp>"
        self.assertEqual (repr (sf), text)
        sf = video.Surface (0, 0, 16)
        text = "<Surface 0x0@16bpp>"
        self.assertEqual (repr (sf), text)
        sf = video.Surface (34, 728, 24)
        text = "<Surface 34x728@24bpp>"
        self.assertEqual (repr (sf), text)
        sf = video.Surface (1, 1, 32)
        text = "<Surface 1x1@32bpp>"
        self.assertEqual (repr (sf), text)
        video.quit ()

if __name__ == "__main__":
    unittest.main ()
