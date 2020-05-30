# -*- coding: utf-8 -*-

import unittest
import os

import pygame, pygame.transform
from pygame.compat import unicode_

from pygame import display


SDL2 = pygame.get_sdl_version()[0] >= 2


class DisplayModuleTest(unittest.TestCase):
    default_caption = "pygame window"

    def setUp(self):
        display.init()

    def tearDown(self):
        display.quit()

    def test_update(self):
        """ see if pygame.display.update takes rects with negative values.
            "|Tags:display|"
        """
        screen = pygame.display.set_mode((100, 100))
        screen.fill((55, 55, 55))

        r1 = pygame.Rect(0, 0, 100, 100)
        pygame.display.update(r1)

        r2 = pygame.Rect(-10, 0, 100, 100)
        pygame.display.update(r2)

        r3 = pygame.Rect(-10, 0, -100, -100)
        pygame.display.update(r3)

    def test_Info(self):
        inf = pygame.display.Info()
        self.assertNotEqual(inf.current_h, -1)
        self.assertNotEqual(inf.current_w, -1)
        # probably have an older SDL than 1.2.10 if -1.

        screen = pygame.display.set_mode((128, 128))
        inf = pygame.display.Info()
        self.assertEqual(inf.current_h, 128)
        self.assertEqual(inf.current_w, 128)

    def todo_test_flip(self):

        # __doc__ (as of 2008-08-02) for pygame.display.flip:

        # pygame.display.flip(): return None
        # update the full display Surface to the screen
        #
        # This will update the contents of the entire display. If your display
        # mode is using the flags pygame.HWSURFACE and pygame.DOUBLEBUF, this
        # will wait for a vertical retrace and swap the surfaces. If you are
        # using a different type of display mode, it will simply update the
        # entire contents of the surface.
        #
        # When using an pygame.OPENGL display mode this will perform a gl buffer swap.

        self.fail()

    def todo_test_get_active(self):

        # __doc__ (as of 2008-08-02) for pygame.display.get_active:

        # pygame.display.get_active(): return bool
        # true when the display is active on the display
        #
        # After pygame.display.set_mode() is called the display Surface will
        # be visible on the screen. Most windowed displays can be hidden by
        # the user. If the display Surface is hidden or iconified this will
        # return False.
        #

        self.fail()

    def test_get_caption(self):
        screen = display.set_mode((100, 100))

        self.assertEqual(display.get_caption()[0], self.default_caption)

    def test_set_caption(self):
        TEST_CAPTION = "test"
        screen = display.set_mode((100, 100))

        self.assertIsNone(display.set_caption(TEST_CAPTION))
        self.assertEqual(display.get_caption()[0], TEST_CAPTION)
        self.assertEqual(display.get_caption()[1], TEST_CAPTION)

    def test_caption_unicode(self):
        TEST_CAPTION = u"台"
        display.set_caption(TEST_CAPTION)
        import sys

        if sys.version_info.major >= 3:
            self.assertEqual(display.get_caption()[0], TEST_CAPTION)
        else:
            self.assertEqual(unicode_(display.get_caption()[0], "utf8"), TEST_CAPTION)

    def todo_test_get_driver(self):

        # __doc__ (as of 2008-08-02) for pygame.display.get_driver:

        # pygame.display.get_driver(): return name
        # get the name of the pygame display backend
        #
        # Pygame chooses one of many available display backends when it is
        # initialized. This returns the internal name used for the display
        # backend. This can be used to provide limited information about what
        # display capabilities might be accelerated. See the SDL_VIDEODRIVER
        # flags in pygame.display.set_mode() to see some of the common
        # options.
        #

        self.fail()

    def test_get_init(self):
        """Ensures the module's initialization state can be retrieved."""
        # display.init() already called in setUp()
        self.assertTrue(display.get_init())

    # This decorator can be removed (or test changed) when issues #991 and #993
    # are resolved.
    @unittest.skipIf(SDL2, "SDL2 issues")
    def test_get_surface(self):
        """Ensures get_surface gets the current display surface."""
        lengths = (1, 5, 100)

        for expected_size in ((w, h) for w in lengths for h in lengths):
            for expected_depth in (8, 16, 24, 32):
                expected_surface = display.set_mode(expected_size, 0, expected_depth)

                surface = pygame.display.get_surface()

                self.assertEqual(surface, expected_surface)
                self.assertIsInstance(surface, pygame.Surface)
                self.assertEqual(surface.get_size(), expected_size)
                self.assertEqual(surface.get_bitsize(), expected_depth)

    def test_get_surface__mode_not_set(self):
        """Ensures get_surface handles the display mode not being set."""
        surface = pygame.display.get_surface()

        self.assertIsNone(surface)

    def todo_test_get_wm_info(self):

        # __doc__ (as of 2008-08-02) for pygame.display.get_wm_info:

        # pygame.display.get_wm_info(): return dict
        # Get information about the current windowing system
        #
        # Creates a dictionary filled with string keys. The strings and values
        # are arbitrarily created by the system. Some systems may have no
        # information and an empty dictionary will be returned. Most platforms
        # will return a "window" key with the value set to the system id for
        # the current display.
        #
        # New with pygame 1.7.1

        self.fail()

    def todo_test_gl_get_attribute(self):

        # __doc__ (as of 2008-08-02) for pygame.display.gl_get_attribute:

        # pygame.display.gl_get_attribute(flag): return value
        # get the value for an opengl flag for the current display
        #
        # After calling pygame.display.set_mode() with the pygame.OPENGL flag,
        # it is a good idea to check the value of any requested OpenGL
        # attributes. See pygame.display.gl_set_attribute() for a list of
        # valid flags.
        #

        self.fail()

    def todo_test_gl_set_attribute(self):

        # __doc__ (as of 2008-08-02) for pygame.display.gl_set_attribute:

        # pygame.display.gl_set_attribute(flag, value): return None
        # request an opengl display attribute for the display mode
        #
        # When calling pygame.display.set_mode() with the pygame.OPENGL flag,
        # Pygame automatically handles setting the OpenGL attributes like
        # color and doublebuffering. OpenGL offers several other attributes
        # you may want control over. Pass one of these attributes as the flag,
        # and its appropriate value. This must be called before
        # pygame.display.set_mode()
        #
        # The OPENGL flags are;
        #   GL_ALPHA_SIZE, GL_DEPTH_SIZE, GL_STENCIL_SIZE, GL_ACCUM_RED_SIZE,
        #   GL_ACCUM_GREEN_SIZE,  GL_ACCUM_BLUE_SIZE, GL_ACCUM_ALPHA_SIZE,
        #   GL_MULTISAMPLEBUFFERS, GL_MULTISAMPLESAMPLES, GL_STEREO

        self.fail()

    def todo_test_iconify(self):

        # __doc__ (as of 2008-08-02) for pygame.display.iconify:

        # pygame.display.iconify(): return bool
        # iconify the display surface
        #
        # Request the window for the display surface be iconified or hidden.
        # Not all systems and displays support an iconified display. The
        # function will return True if successfull.
        #
        # When the display is iconified pygame.display.get_active() will
        # return False. The event queue should receive a ACTIVEEVENT event
        # when the window has been iconified.
        #

        self.fail()

    def test_init(self):
        """Ensures the module is initialized after init called."""
        # display.init() already called in setUp(), so quit and re-init
        display.quit()
        display.init()

        self.assertTrue(display.get_init())

    def test_init__multiple(self):
        """Ensures the module is initialized after multiple init calls."""
        display.init()
        display.init()

        self.assertTrue(display.get_init())

    def test_list_modes(self):
        modes = pygame.display.list_modes(depth=0, flags=pygame.FULLSCREEN, display=0)
        # modes == -1 means any mode is supported.
        if modes != -1:
            self.assertEqual(len(modes[0]), 2)
            self.assertEqual(type(modes[0][0]), int)

        modes = pygame.display.list_modes()
        if modes != -1:
            self.assertEqual(len(modes[0]), 2)
            self.assertEqual(type(modes[0][0]), int)

        modes = pygame.display.list_modes(depth=0, flags=0, display=0)
        if modes != -1:
            self.assertEqual(len(modes[0]), 2)
            self.assertEqual(type(modes[0][0]), int)

    def test_mode_ok(self):
        pygame.display.mode_ok((128, 128))
        modes = pygame.display.list_modes()
        if modes != -1:
            size = modes[0]
            self.assertNotEqual(pygame.display.mode_ok(size), 0)

        pygame.display.mode_ok((128, 128), 0, 32)
        pygame.display.mode_ok((128, 128), flags=0, depth=32, display=0)

    def test_mode_ok_fullscreen(self):
        modes = pygame.display.list_modes()
        if modes != -1:
            size = modes[0]
            self.assertNotEqual(
                pygame.display.mode_ok(size, flags=pygame.FULLSCREEN), 0
            )

    def test_mode_ok_scaled(self):
        modes = pygame.display.list_modes()
        if modes != -1:
            size = modes[0]
            self.assertNotEqual(pygame.display.mode_ok(size, flags=pygame.SCALED), 0)

    def test_get_num_displays(self):
        self.assertGreater(pygame.display.get_num_displays(), 0)

    def test_quit(self):
        """Ensures the module is not initialized after quit called."""
        display.quit()

        self.assertFalse(display.get_init())

    def test_quit__multiple(self):
        """Ensures the module is not initialized after multiple quit calls."""
        display.quit()
        display.quit()

        self.assertFalse(display.get_init())

    def todo_test_set_gamma(self):

        # __doc__ (as of 2008-08-02) for pygame.display.set_gamma:

        # pygame.display.set_gamma(red, green=None, blue=None): return bool
        # change the hardware gamma ramps
        #
        # Set the red, green, and blue gamma values on the display hardware.
        # If the green and blue arguments are not passed, they will both be
        # the same as red. Not all systems and hardware support gamma ramps,
        # if the function succeeds it will return True.
        #
        # A gamma value of 1.0 creates a linear color table. Lower values will
        # darken the display and higher values will brighten.
        #

        self.fail()

    def todo_test_set_gamma_ramp(self):

        # __doc__ (as of 2008-08-02) for pygame.display.set_gamma_ramp:

        # change the hardware gamma ramps with a custom lookup
        # pygame.display.set_gamma_ramp(red, green, blue): return bool
        # set_gamma_ramp(red, green, blue): return bool
        #
        # Set the red, green, and blue gamma ramps with an explicit lookup
        # table. Each argument should be sequence of 256 integers. The
        # integers should range between 0 and 0xffff. Not all systems and
        # hardware support gamma ramps, if the function succeeds it will
        # return True.
        #

        self.fail()

    def todo_test_set_icon(self):

        # __doc__ (as of 2008-08-02) for pygame.display.set_icon:

        # pygame.display.set_icon(Surface): return None
        # change the system image for the display window
        #
        # Sets the runtime icon the system will use to represent the display
        # window. All windows default to a simple pygame logo for the window
        # icon.
        #
        # You can pass any surface, but most systems want a smaller image
        # around 32x32. The image can have colorkey transparency which will be
        # passed to the system.
        #
        # Some systems do not allow the window icon to change after it has
        # been shown. This function can be called before
        # pygame.display.set_mode() to create the icon before the display mode
        # is set.
        #

        self.fail()

    def test_set_mode_kwargs(self):

        pygame.display.set_mode(size=(1, 1), flags=0, depth=0, display=0)

    def test_set_mode_scaled(self):
        surf = pygame.display.set_mode(
            size=(1, 1), flags=pygame.SCALED, depth=0, display=0
        )
        winsize = pygame.display.get_window_size()
        self.assertEqual(
            winsize[0] % surf.get_size()[0],
            0,
            "window width should be a multiple of the surface width",
        )
        self.assertEqual(
            winsize[1] % surf.get_size()[1],
            0,
            "window height should be a multiple of the surface height",
        )
        self.assertEqual(
            winsize[0] / surf.get_size()[0], winsize[1] / surf.get_size()[1]
        )

    def test_screensaver_support(self):
        pygame.display.set_allow_screensaver(True)
        self.assertTrue(pygame.display.get_allow_screensaver())
        pygame.display.set_allow_screensaver(False)
        self.assertFalse(pygame.display.get_allow_screensaver())
        pygame.display.set_allow_screensaver()
        self.assertTrue(pygame.display.get_allow_screensaver())


    def todo_test_set_palette(self):

        # __doc__ (as of 2008-08-02) for pygame.display.set_palette:

        # pygame.display.set_palette(palette=None): return None
        # set the display color palette for indexed displays
        #
        # This will change the video display color palette for 8bit displays.
        # This does not change the palette for the actual display Surface,
        # only the palette that is used to display the Surface. If no palette
        # argument is passed, the system default palette will be restored. The
        # palette is a sequence of RGB triplets.
        #

        self.fail()

    def todo_test_toggle_fullscreen(self):

        # __doc__ (as of 2008-08-02) for pygame.display.toggle_fullscreen:

        # pygame.display.toggle_fullscreen(): return bool
        # switch between fullscreen and windowed displays
        #
        # Switches the display window between windowed and fullscreen modes.
        # This function only works under the unix x11 video driver. For most
        # situations it is better to call pygame.display.set_mode() with new
        # display flags.
        #

        self.fail()


@unittest.skipIf(
    os.environ.get("SDL_VIDEODRIVER") == "dummy",
    'OpenGL requires a non-"dummy" SDL_VIDEODRIVER',
)
class DisplayOpenGLTest(unittest.TestCase):
    def test_screen_size_opengl(self):
        """ returns a surface with the same size requested.
        |tags:display,slow,opengl|
        """
        pygame.display.init()
        screen = pygame.display.set_mode((640, 480), pygame.OPENGL)
        self.assertEqual((640, 480), screen.get_size())


class X11CrashTest(unittest.TestCase):
    def test_x11_set_mode_crash_gh1654(self):
        # Test for https://github.com/pygame/pygame/issues/1654
        # If unfixed, this will trip a segmentation fault
        pygame.display.init()
        pygame.display.quit()
        screen = pygame.display.set_mode((640, 480), 0)
        self.assertEqual((640, 480), screen.get_size())


if __name__ == "__main__":
    unittest.main()
