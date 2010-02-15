import os
try:
    import pygame2.test.pgunittest as unittest
    from pygame2.test.pgunittest import doprint, interactive
except:
    import pgunittest as unittest
    from pgunittest import doprint, interactive

import pygame2
import pygame2.sdl.wm as wm
import pygame2.sdl.video as video
import pygame2.sdl.constants as constants

class SDLWMTest (unittest.TestCase):

    def test_pygame2_sdl_wm_get_caption(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.wm.get_caption:

        # get_caption () -> str
        # 
        # Gets the caption of the current SDL window.
        self.assertRaises (pygame2.Error, wm.get_caption)
        video.init ()
        self.assertRaises (pygame2.Error, wm.get_caption)
        sf = video.set_mode (10, 10)
        self.assertEqual (wm.get_caption (), (None, None))
        wm.set_caption ("test window")
        self.assertEqual (wm.get_caption (), ("test window", None))
        wm.set_caption ("", "icon")
        self.assertEqual (wm.get_caption (), ("", "icon"))
        wm.set_caption ("test window", "icon")
        self.assertEqual (wm.get_caption (), ("test window", "icon"))
        video.quit ()

    def todo_test_pygame2_sdl_wm_get_info(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.wm.get_info:

        # get_info () -> dict
        # 
        # Gets operating system and window manager specific information
        # about the current SDL window .

        self.fail() 

    @interactive ("Does the window have the input focus now?")
    def todo_test_pygame2_sdl_wm_grab_input(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.wm.grab_input:

        # grab_input (mode) -> int
        # 
        # Advices the window manager to let the window grab the mouse
        # and keyboard input focus (or release it). The function returns
        # the previously set mode.
        doprint ("Trying to grab mouse and keyboard input focus.")
        self.fail() 

    @interactive ("Was the window iconified (if supported by the WM)?")
    def todo_test_pygame2_sdl_wm_iconify_window(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.wm.iconify_window:

        # iconify_window () -> bool
        # 
        # Advices the window manager to iconify the window and returns,
        # whether the operation was successful or not.
        doprint ("Trying to iconify window...")

        self.fail() 

    def test_pygame2_sdl_wm_set_caption(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.wm.set_caption:

        # set_caption (str) -> None
        # 
        # Sets the caption of the current SDL window.

        # handled in wm_get_caption()
        self.assertRaises (pygame2.Error, wm.set_caption, "test", "test")
        video.init ()
        self.assertRaises (pygame2.Error, wm.set_caption, "test", "test")
        sf = video.set_mode (10, 10)
        self.assertEqual (wm.get_caption (), (None, None))
        self.assert_ (wm.set_caption ("test window") == None)
        self.assertEqual (wm.get_caption (), ("test window", None))
        self.assert_ (wm.set_caption ("", "icon") == None)
        self.assertEqual (wm.get_caption (), ("", "icon"))
        self.assert_ (wm.set_caption ("test window", "icon") == None)
        self.assertEqual (wm.get_caption (), ("test window", "icon"))
        video.quit ()

    @interactive ("Was the window icon updated correctly?")
    def todo_test_pygame2_sdl_wm_set_icon(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.wm.set_icon:

        # set_icon (surface[, mask]) -> None
        # 
        # Sets the window manager icon to be used by the window.
        # 
        # .. todo
        # 
        # Not fully implemented.
        doprint ("Trying to change the window icon to a red ball...")

        self.fail()

    @interactive ("Was the fullscreen mode correctly toggled?")
    def todo_test_pygame2_sdl_wm_toggle_fullscreen(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.wm.toggle_fullscreen:

        # toggle_fullscren (bool) -> bool
        # 
        # Tries to toggle the fullscreen mode for the SDL window and
        # returns, whether the operation was successful or not.
        ptext = "Trying to switch to fullscreen mode." + os.linesep + \
                "It will be switched back after 5 seconds."
        self.fail()

if __name__ == "__main__":
    unittest.main ()
