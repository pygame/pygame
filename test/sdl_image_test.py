import os, sys
import unittest
try:
    import StringIO as stringio
except ImportError:
    import io as stringio

import pygame2
import pygame2.sdl.image as image
import pygame2.sdl.video as video
import pygame2.sdl.constants as constants

class SDLImageTest (unittest.TestCase):
    def test_pygame2_sdl_image_load_bmp(self):

        # __doc__ (as of 2009-05-14) for pygame2.sdl.image.load_bmp:

        # load_bmp (file) -> pygame2.sdl.video.Surface
        # 
        # Loads a BMP file and creates a pygame2.sdl.video.Surface from it.
        # 
        # load_bmp (file) -> pygame2.sdl.video.Surface  Loads a BMP file and
        # creates a pygame2.sdl.video.Surface from it.  Loads a BMP file and
        # creates a pygame2.sdl.video.Surface from it. The file argument can
        # be either a file object or the filename.
        video.init ()
        imgdir = os.path.dirname (os.path.abspath (__file__))
        sf = image.load_bmp (os.path.join (imgdir, "test.bmp"))
        self.assertEqual (sf.size, (16, 16))
        video.quit ()

    def test_pygame2_sdl_image_save_bmp(self):

        # __doc__ (as of 2009-05-14) for pygame2.sdl.image.save_bmp:

        # save_bmp (surface, file) -> None
        # 
        # Saves a surface to a bitmap file.
        # 
        # save_bmp (surface, file) -> None  Saves a surface to a bitmap file.
        # Saves a pygame2.sdl.video.Surface to the specified file, where file
        # can be a filename or file object.
        video.init ()
        imgdir = os.path.dirname (os.path.abspath (__file__))
        sf = image.load_bmp (os.path.join (imgdir, "test.bmp"))
        buf = None
        if sys.version_info[0] >= 3:
            buf = stringio.BytesIO ()
        else:
            buf = stringio.StringIO ()
        self.assert_ (image.save_bmp (sf, buf) == None)
        self.assertEqual (os.stat (os.path.join (imgdir, "test.bmp")).st_size,
                          len (buf.getvalue ()))
        video.quit ()

if __name__ == "__main__":
    unittest.main ()
