import sys
import unittest
import pygame2
from pygame2.colorpalettes import CGAPALETTE
from pygame2 import Rect, Color
import pygame2.sdl.video as video
import pygame2.sdl.constants as constants

class SDLVideoPixelFormatTest (unittest.TestCase):

    def test_pygame2_sdl_video_PixelFormat_alpha(self):

        # __doc__ (as of 2009-12-08) for pygame2.sdl.video.PixelFormat.alpha:

        # Gets or sets the overall surface alpha value in the
        # range [0, 255].

        def _seta (fmt, a):
            fmt.alpha = a
        
        format = video.PixelFormat ()
        self.assertEqual (format.alpha, 0)
        
        self.assertRaises (ValueError, _seta, format, -5)
        self.assertRaises (ValueError, _seta, format, -1)
        self.assertRaises (ValueError, _seta, format, 256)
        format.alpha = 255
        self.assertEqual (format.alpha, 255)
        format.alpha = 127
        self.assertEqual (format.alpha, 127)
        format.alpha = 33
        self.assertEqual (format.alpha, 33)
        for i in range (0, 255):
            format.alpha = i
            self.assertEqual (format.alpha, i)

    def test_pygame2_sdl_video_PixelFormat_bits_per_pixel(self):

        # __doc__ (as of 2009-12-08) for pygame2.sdl.video.PixelFormat.bits_per_pixel:

        # Gets or sets the bits per pixel to use for
        # storing a RGBA value.
        
        def _setbpp (fmt, bpp):
            fmt.bits_per_pixel = bpp
        
        format = video.PixelFormat ()
        self.assertEqual (format.bits_per_pixel, 0)
        self.assertEqual (format.bytes_per_pixel, 0)
        
        self.assertRaises (ValueError, _setbpp, format, -5)
        self.assertRaises (ValueError, _setbpp, format, -1)
        self.assertRaises (ValueError, _setbpp, format, 256)
        for i in range (0, 255):
            format.bits_per_pixel = i
            self.assertEqual (format.bytes_per_pixel, 0)
            self.assertEqual (format.bits_per_pixel, i)

    def test_pygame2_sdl_video_PixelFormat_bytes_per_pixel(self):

        # __doc__ (as of 2009-12-08) for pygame2.sdl.video.PixelFormat.bytes_per_pixel:

        # Gets or sets the bytes per pixel to use for
        # storing a RGBA value.

        def _setbpp (fmt, bpp):
            fmt.bytes_per_pixel = bpp
        
        format = video.PixelFormat ()
        self.assertEqual (format.bytes_per_pixel, 0)
        self.assertEqual (format.bits_per_pixel, 0)
        
        self.assertRaises (ValueError, _setbpp, format, -5)
        self.assertRaises (ValueError, _setbpp, format, -1)
        self.assertRaises (ValueError, _setbpp, format, 256)
        for i in range (0, 255):
            format.bytes_per_pixel = i
            self.assertEqual (format.bytes_per_pixel, i)
            self.assertEqual (format.bits_per_pixel, 0)

    def test_pygame2_sdl_video_PixelFormat_colorkey(self):

        # __doc__ (as of 2009-12-08) for pygame2.sdl.video.PixelFormat.colorkey:

        # Gets or sets the RGBA value of transparent pixels.

        def _setcolorkey (fmt, key):
            fmt.colorkey = key
        
        format = video.PixelFormat ()
        self.assertEqual (format.colorkey, Color (0x00000000))

        self.assertRaises (ValueError, _setcolorkey, format, -200)
        self.assertRaises (ValueError, _setcolorkey, format, -1)
        self.assertRaises (ValueError, _setcolorkey, format, 200)
        self.assertRaises (ValueError, _setcolorkey, format, "hello")
        self.assertRaises (ValueError, _setcolorkey, format, None)

        # TODO: test assignments
        
    def todo_test_pygame2_sdl_video_PixelFormat_get_rgba(self):

        # __doc__ (as of 2009-12-08) for pygame2.sdl.video.PixelFormat.get_rgba:

        # get_rgba (color) -> pygame2.Color
        # 
        # Gets the best matching pygame2.Color for the passed color.
        # 
        # Gets a color value, which fits the PixelFormat best. This means
        # that an internal conversion is done (on demand) to match the
        # passed color to the PixelFormat's supported value ranges. If
        # the PixelFormat does not have alpha transparency support, the
        # color's alpha value will be set to fully opaque (255).

        self.fail() 

    def todo_test_pygame2_sdl_video_PixelFormat_losses(self):

        # __doc__ (as of 2009-12-08) for pygame2.sdl.video.PixelFormat.losses:

        # Gets or sets the precision loss of each RGBA color
        # component.

        self.fail() 

    def todo_test_pygame2_sdl_video_PixelFormat_map_rgba(self):

        # __doc__ (as of 2009-12-08) for pygame2.sdl.video.PixelFormat.map_rgba:

        # map_rgba (r, g, b[, a]) -> long
        # map_rgba (color) -> long
        # 
        # Converts a color to the best value matching the format.
        # 
        # Gets a color value, which fits the PixelFormat best. This means
        # that an internal conversion is done (on demand) to match the
        # passed color to the PixelFormat's supported value ranges.
        # Instead of returning a color as in get_rgba, an integer value
        # matching the PixelFormat's supported value ranges will be
        # returned.

        self.fail() 

    def todo_test_pygame2_sdl_video_PixelFormat_masks(self):

        # __doc__ (as of 2009-12-08) for pygame2.sdl.video.PixelFormat.masks:

        # Gets or sets the binary masks used to retrieve individual
        # color values.

        self.fail() 

    def todo_test_pygame2_sdl_video_PixelFormat_palette(self):

        # __doc__ (as of 2009-12-08) for pygame2.sdl.video.PixelFormat.palette:

        # Gets the color palette (if any) used by the PixelFormat. If the
        # PixelFormat does not have any palette, None will be returned.

        self.fail() 

    def test_pygame2_sdl_video_PixelFormat_readonly(self):

        # __doc__ (as of 2009-12-08) for pygame2.sdl.video.PixelFormat.readonly:

        # Gets, whether the PixelFormat is read-only (this cannot be
        # changed).
        video.init ()
        
        format = video.PixelFormat ()
        self.assertEqual (format.readonly, False)

        surface = video.Surface (1, 1)
        fmt = surface.format
        self.assertEqual (fmt.readonly, True)
        
        def _setr (format, readonly):
            format.readonly = readonly

        if sys.version_info < (2, 5):
            self.assertRaises (TypeError, _setr, format, True)
            self.assertRaises (TypeError, _setr, format, False)
        else:
            self.assertRaises (AttributeError, _setr, format, True)
            self.assertRaises (AttributeError, _setr, format, False)
        
        video.quit ()

    def todo_test_pygame2_sdl_video_PixelFormat_shifts(self):

        # __doc__ (as of 2009-12-08) for pygame2.sdl.video.PixelFormat.shifts:

        # Gets the binary left shift of each color component in
        # the pixel value.

        self.fail() 

if __name__ == "__main__":
    unittest.main ()
