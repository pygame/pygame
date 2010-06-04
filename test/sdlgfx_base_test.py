import unittest
import time
import pygame2
import pygame2.sdlgfx.base as base
import pygame2.sdlgfx.constants as constants

class SDLGfxTest (unittest.TestCase):

    def test_pygame2_sdlgfx_base_FPSmanager (self):
        manager = base.FPSmanager ()
        self.assertEqual (type (manager), base.FPSmanager)
        self.assertEqual (manager.framerate, constants.FPS_DEFAULT)

        self.assertRaises (ValueError, base.FPSmanager, 0)
        self.assertRaises (ValueError, base.FPSmanager, -10)
        self.assertRaises (ValueError, base.FPSmanager, 201)
        self.assertRaises (ValueError, base.FPSmanager, 400)

    def test_pygame2_sdlgfx_base_FPSmanager_delay(self):

        # __doc__ (as of 2010-01-06) for pygame2.sdlgfx.base.FPSmanager.delay:

        # delay () -> None
        # 
        # Delays the execution of the application to keep up the desired frame
        # rate.
        for i in range (10):
            prev = time.time ()
            manager = base.FPSmanager (10)
            manager.delay ()
            last = time.time ()
            delaytime = (last - prev) * 1000
            self.assert_ (85 < delaytime < 115)

    def test_pygame2_sdlgfx_base_FPSmanager_framerate(self):

        # __doc__ (as of 2010-01-06) for pygame2.sdlgfx.base.FPSmanager.framerate:

        # Gets or sets the frame rate to keep.
        self.assertEqual (base.FPSmanager (constants.FPS_LOWER_LIMIT).framerate,
                          constants.FPS_LOWER_LIMIT)
        self.assertEqual (base.FPSmanager (constants.FPS_UPPER_LIMIT).framerate,
                          constants.FPS_UPPER_LIMIT)

    def test_pygame2_sdlgfx_base_get_compiled_version(self):

        # __doc__ (as of 2010-01-06) for pygame2.sdlgfx.base.get_compiled_version:

        # get_compiled_version () -> (int, int, int)
        # 
        # Gets the SDL_gfx version pygame2 was compiled against as
        # three-value tuple.
        # 
        # This version is built at compile time. It can be used to detect
        # which features may not be available through Pygame, if it is used
        # as precompiled package using a different version of the SDL_gfx
        # library.
        self.assertEqual (len (base.get_compiled_version ()), 3)
        self.assertEqual (base.get_compiled_version ()[0], 2)
        self.assertEqual (base.get_compiled_version ()[1], 0)
        self.assert_ (base.get_compiled_version ()[2] >= 18)

if __name__ == "__main__":
    unittest.main ()
