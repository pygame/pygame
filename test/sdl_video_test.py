import unittest
import pygame2
import pygame2.sdl.video as video
import pygame2.sdl.constants as constants

class SDLVideoTest (unittest.TestCase):
    __tags__ = [ "sdl" ]

    def test_pygame2_sdl_video_get_drivername(self):

        # __doc__ (as of 2009-05-31) for pygame2.sdl.video.get_drivername:

        # get_drivername () -> str
        # 
        # Gets the name of the video driver.
        # 
        # Gets the name of the video driver or None, if the video system has
        # not been initialised or it could not be determined.
        self.assert_ (video.get_drivername () == None)
        video.init ()
        self.assert_ (video.get_drivername () != None)
        video.quit ()
        self.assert_ (video.get_drivername () == None)

    def test_pygame2_sdl_video_get_gammaramp(self):

        # __doc__ (as of 2009-05-31) for pygame2.sdl.video.get_gammaramp:

        # get_gammaramp () -> (int, int, ...), (int, int, ...), (int, int, ...)
        # 
        # Gets the color gamma lookup tables for the display.
        # 
        # Gets the color gamma lookup table for the display. This will
        # return three tuples for the red, green and blue gamma values. Each
        # tuple contains 256 values.
        video.init ()
        r, g, b = video.get_gammaramp ()
        self.assert_ (len (r) == 256)
        self.assert_ (len (g) == 256)
        self.assert_ (len (b) == 256)
        for v in r:
            self.assert_ (type (v) == int)
        for v in g:
            self.assert_ (type (v) == int)
        for v in b:
            self.assert_ (type (v) == int)
        video.quit ()

    def test_pygame2_sdl_video_get_info(self):

        # __doc__ (as of 2009-05-31) for pygame2.sdl.video.get_info:

        # get_info () -> dict
        # 
        # Gets information about the video hardware.
        # 
        # Gets information about the video hardware. The returned dictionary
        # contains the following entries.
        # 
        # +------------------+---------------------------------------------+
        # | Entry            | Meaning                                     |
        # +==================+=============================================+
        # | hw_available     | Is it possible to create hardware surfaces? |
        # +------------------+---------------------------------------------+
        # | wm_available     | Is a window manager available?              |
        # +------------------+---------------------------------------------+
        # | blit_hw          | Are hardware to hardware blits accelerated? |
        # +------------------+---------------------------------------------+
        # | blit_hw_CC       | Are hardware to hardware colorkey blits     |
        # |                  | accelerated?                                |
        # +------------------+---------------------------------------------+
        # | blit_hw_A        | Are hardware to hardware alpha blits        |
        # |                  | accelerated?                                |
        # +------------------+---------------------------------------------+
        # | blit_sw          | Are software to hardware blits accelerated? |
        # +------------------+---------------------------------------------+
        # | blit_sw_CC       | Are software to hardware colorkey blits     |
        # |                  | accelerated?                                |
        # +------------------+---------------------------------------------+
        # | blit_sw_A        | Are software to hardware alpha blits        |
        # |                  | accelerated?                                |
        # +------------------+---------------------------------------------+
        # | blit_fill        | Are color fills accelerated?                |
        # +------------------+---------------------------------------------+
        # | video_mem        | Total amount of video memory in Kilobytes   |
        # +------------------+---------------------------------------------+
        # | vfmt             | Pixel format of the video device            |
        # +------------------+---------------------------------------------+
        self.assertRaises (pygame2.Error, video.get_info)
        video.init ()
        info = video.get_info ()
        self.assert_ (type (info) == dict)
        self.assert_ (type (info["hw_available"]) == bool)
        self.assert_ (type (info["wm_available"]) == bool)
        self.assert_ (type (info["blit_hw"]) == bool)
        self.assert_ (type (info["blit_hw_CC"]) == bool)
        self.assert_ (type (info["blit_hw_A"]) == bool)
        self.assert_ (type (info["blit_sw"]) == bool)
        self.assert_ (type (info["blit_sw_CC"]) == bool)
        self.assert_ (type (info["blit_sw_A"]) == bool)
        self.assert_ (type (info["blit_fill"]) == bool)
        self.assert_ (type (info["video_mem"]) == int)
        self.assert_ (type (info["vfmt"]) == video.PixelFormat)
        video.quit ()

    def test_pygame2_sdl_video_get_videosurface(self):

        # __doc__ (as of 2009-05-31) for pygame2.sdl.video.get_videosurface:

        # get_videosurface () -> Surface
        # 
        # Gets the current display surface or None, if there is no such Surface.
        self.assertRaises (pygame2.Error, video.get_videosurface)
        video.init ()
        self.assert_ (video.get_videosurface () == None)
        video.quit ()
        self.assertRaises (pygame2.Error, video.get_videosurface)
        video.init ()
        video.set_mode (1, 1)
        self.assert_ (type (video.get_videosurface ()) == video.Surface)
        video.quit ()
        self.assertRaises (pygame2.Error, video.get_videosurface)

    def test_pygame2_sdl_video_init(self):

        # __doc__ (as of 2009-05-31) for pygame2.sdl.video.init:

        # init () -> None
        # 
        # Initializes the video subsystem of the SDL library.
        self.assertEqual (video.init (), None)
        video.quit ()

    def test_pygame2_sdl_video_is_mode_ok(self):

        # __doc__ (as of 2009-05-31) for pygame2.sdl.video.is_mode_ok:

        # is_mode_ok (width, height[, bpp, flags]) -> bool
        # is_mode_ok (size[, bpp, flags]) -> bool
        # 
        # Checks, whether the requested video mode is supported.
        # 
        # Checks, whether the video mode is supported for the passed size,
        # bit depth and flags. If the bit depth (bpp) argument is omitted,
        # the current screen bit depth will be used.
        # 
        # The optional flags argument is the same as for set_mode.
        self.assertRaises (pygame2.Error, video.is_mode_ok)
        video.init ()
        modes = video.list_modes ()
        for r in modes:
            self.assert_ (video.is_mode_ok (r.size) == True)
        video.quit ()
        self.assertRaises (pygame2.Error, video.is_mode_ok)

    def test_pygame2_sdl_video_list_modes(self):

        # __doc__ (as of 2009-05-31) for pygame2.sdl.video.list_modes:

        # list_modes ([, format, flags]) -> [rect, rect, ...]
        # 
        # Returns the supported modes for a specific format and flags.
        # 
        # Returns the supported modes for a specific format and flags.
        # The optional format argument must be a PixelFormat
        # instance with the desired mode information. The optional flags
        # argument is the same as for set_mode.
        # 
        # If both, the format and flags are omitted, all supported screen
        # resolutions for all supported formats and flags are returned.
        self.assertRaises (pygame2.Error, video.list_modes)
        video.init ()
        modes = video.list_modes()
        self.assert_ (type (modes) == list)
        for r in modes:
            self.assert_ (type (r) == pygame2.Rect)
        video.quit ()
        self.assertRaises (pygame2.Error, video.list_modes)

    def test_pygame2_sdl_video_quit(self):

        # __doc__ (as of 2009-05-31) for pygame2.sdl.video.quit:

        # quit () -> None
        # 
        # Shuts down the video subsystem of the SDL library.
        # 
        # After calling this function, you should not invoke any class,
        # method or function related to the video subsystem as they are
        # likely to fail or might give unpredictable results.
        self.assertEqual (video.quit (), None)

    def test_pygame2_sdl_video_set_gamma(self):

        # __doc__ (as of 2009-05-31) for pygame2.sdl.video.set_gamma:

        # set_gamma (red, green, blue) -> None
        # 
        # Sets the gamma values for all three color channels.
        # 
        # Sets the gamma values for all three color channels. In case
        # adjusting the gamma is not supported, an exception will be raised.
        video.init ()
        try:
            video.set_gamma (1, 1, 1)
        except pygame2.Error:
            video.quit ()
            return
        self.assert_ (video.set_gamma (0, 0, 0) == None)
        self.assert_ (video.set_gamma (1, 1, 1) == None)
        self.assert_ (video.set_gamma (10, 1, 1) == None)
        self.assert_ (video.set_gamma (10, 12.88, 3.385) == None)
        self.assert_ (video.set_gamma (-10, -12.88, -3.385) == None)
        video.quit ()

    def todo_test_pygame2_sdl_video_set_gammaramp(self):

        # __doc__ (as of 2009-05-31) for pygame2.sdl.video.set_gammaramp:

        # set_gammaramp (redtable, greentable, bluetable) -> None
        # 
        # Sets the color gamma lookup tables for the display.
        # 
        # Sets the color gamma lookup table for the display. The three
        # arguments must be sequences with 256 integer value enties for the
        # gamma ramps.

        self.fail() 

    def todo_test_pygame2_sdl_video_set_mode(self):

        # __doc__ (as of 2009-05-31) for pygame2.sdl.video.set_mode:

        # set_mode (width, height[, bpp, flags]) -> Surface
        # set_mode (size[, bpp, flags]) -> Surface
        # 
        # Creates the main display Surface.
        # 
        # Creates the main display Surface using the specified size, bit
        # depth and flags. If the bit depth (bpp) argument is omitted, the
        # current screen bit depth will be used.

        self.fail() 

    def test_pygame2_sdl_video_was_init(self):

        # __doc__ (as of 2009-05-31) for pygame2.sdl.video.was_init:

        # was_init () -> bool
        # 
        # Returns, whether the video subsystem of the SDL library is
        # initialized.
        video.quit ()
        self.assertEqual (video.was_init (), False)
        video.init ()
        self.assertEqual (video.was_init (), True)
        video.quit ()
        self.assertEqual (video.was_init (), False)

if __name__ == "__main__":
    unittest.main ()
