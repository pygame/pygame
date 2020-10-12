# -*- coding: utf-8 -*-

import unittest
import os
import time

import pygame, pygame.transform
from pygame.compat import unicode_
from pygame.locals import *
from pygame.tests.test_utils import question

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

    def test_flip(self):
        screen = pygame.display.set_mode((100, 100))

        #test without a change
        self.assertIsNone(pygame.display.flip())

        #test with a change
        pygame.Surface.fill(screen, (66,66,53))
        self.assertIsNone(pygame.display.flip())

        #test without display init
        pygame.display.quit()
        with self.assertRaises(pygame.error):
            (pygame.display.flip())

        #test without window
        del screen
        with self.assertRaises(pygame.error):
            (pygame.display.flip())

    def test_get_active(self):
        """Test the get_active function"""

        #Initially, the display is not active
        pygame.display.quit()
        self.assertEqual(pygame.display.get_active(), False)

        #get_active defaults to true after a set_mode
        pygame.display.init()
        pygame.display.set_mode((640, 480))
        self.assertEqual(pygame.display.get_active(), True)

        #get_active after init/quit should be False
        #since no display is visible
        pygame.display.quit()
        pygame.display.init()
        self.assertEqual(pygame.display.get_active(), False)


    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") == "dummy",
        'requires the SDL_VIDEODRIVER to be a non dummy value'
    )
    def test_get_active_iconify(self):
        """Test the get_active function after an iconify"""


        #According to the docs, get_active should return
        #false if the display is iconified
        pygame.display.set_mode((640, 480))

        pygame.event.clear()
        pygame.display.iconify()

        for _ in range(100):
            time.sleep(0.01)
            pygame.event.pump()

        self.assertEqual(pygame.display.get_active(), False)

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

    def test_get_driver(self):
        drivers = [
            'aalib',
            'android',
            'arm',
            'cocoa',
            'dga',
            'directx',
            'directfb',
            'dummy',
            'emscripten',
            'fbcon',
            'ggi',
            'haiku',
            'khronos',
            'kmsdrm',
            'nacl',
            'offscreen',
            'pandora',
            'psp',
            'qnx',
            'raspberry',
            'svgalib',
            'uikit',
            'vgl',
            'vivante',
            'wayland',
            'windows',
            'windib',
            'winrt',
            'x11'
        ]
        driver = display.get_driver()
        self.assertIn(driver, drivers)

        display.quit()
        with self.assertRaises(pygame.error):
            driver = display.get_driver()

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

    def test_get_wm_info(self):
        wm_info = display.get_wm_info()
        # Assert function returns a dictionary type
        self.assertIsInstance(wm_info,dict)

        wm_info_potential_keys = {
        'colorbuffer',
        'connection',
        'data',
        'dfb',
        'display',
        'framebuffer',
        'fswindow',
        'hdc',
        'hglrc',
        'hinstance',
        'lock_func',
        'resolveFramebuffer',
        'shell_surface',
        'surface',
        'taskHandle',
        'unlock_func',
        'wimpVersion',
        'window',
        'wmwindow'
        }

        # If any unexpected dict keys are present, they
        # will be stored in set wm_info_remaining_keys
        wm_info_remaining_keys = set(wm_info.keys()).difference(wm_info_potential_keys)

        # Assert set is empty (& therefore does not
        # contain unexpected dict keys)
        self.assertFalse(wm_info_remaining_keys)

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

    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") in ["dummy", "android"],
        'iconify is only supported on some video drivers/platforms'
    )
    def test_iconify(self):
        _ = pygame.display.set_mode((640, 480))

        self.assertEqual(pygame.display.get_active(), True)

        success = pygame.display.iconify()

        if success:
            minimized_event = False
            # make sure we cycle the event loop enough to get the display
            # hidden
            for _ in range(100):
                time.sleep(0.01)
                for event in pygame.event.get():
                    if SDL2:
                        if (event.type == pygame.WINDOWEVENT and
                                event.event == pygame.WINDOWEVENT_MINIMIZED):
                            minimized_event = True

            if SDL2:
                self.assertEqual(minimized_event, True)
                self.assertEqual(pygame.display.get_active(), False)

        else:
            self.fail('Iconify not supported on this platform, please skip')

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

    @unittest.skipIf(
    os.environ.get("SDL_VIDEODRIVER") == "dummy" and not SDL2,
    "Can't set gamma on SDL 1 with the dummy video driver",
    )
    def test_set_gamma(self):
        if(not SDL2):
            pygame.display.set_mode((1, 1))
        gammas = [0.0,0.25,0.5,0.88,1.0]
        for gamma in gammas:
            self.assertEqual(pygame.display.set_gamma(gamma),True)
        gammas = [(0.5,0.5,0.5),(1.0,1.0,1.0),(0.22,0.33,0.44),(0.0,0.0,0.0)]
        for gammaTuple in gammas:
            self.assertEqual(pygame.display.set_gamma(gammaTuple[0],gammaTuple[1],gammaTuple[2]),True)

    @unittest.skipIf(
        not hasattr(pygame.display,"set_gamma_ramp"),
        "Not all systems and hardware support gamma ramps"
    )
    def test_set_gamma_ramp(self):

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
        pygame.display.set_mode((5, 5))
        r = list(range(256))
        g = [number + 256 for number in r]
        b = [number + 256 for number in g]
        isSupported = pygame.display.set_gamma_ramp(r, g, b)
        if (isSupported):
            self.assertTrue(pygame.display.set_gamma_ramp(r, g, b))
        else:
            self.assertFalse(pygame.display.set_gamma_ramp(r, g, b))

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

    @unittest.skipIf(SDL2, "set_palette() not supported in SDL2")
    def test_set_palette(self):
        with self.assertRaises(pygame.error):
            palette = [1, 2, 3]
            pygame.display.set_palette(palette)
        pygame.display.set_mode((1024, 768), 0, 8)
        palette = []
        self.assertIsNone(pygame.display.set_palette(palette))

        with self.assertRaises(ValueError):
            palette = 12
            pygame.display.set_palette(palette)
        with self.assertRaises(TypeError):
            palette = [[1, 2], [1, 2]]
            pygame.display.set_palette(palette)
        with self.assertRaises(TypeError):
            palette = [[0, 0, 0, 0, 0]] + [[x, x, x, x, x]
                                           for x in range(1, 255)]
            pygame.display.set_palette(palette)
        with self.assertRaises(TypeError):
            palette = "qwerty"
            pygame.display.set_palette(palette)
        with self.assertRaises(TypeError):
            palette = [[123, 123, 123]*10000]
            pygame.display.set_palette(palette)
        with self.assertRaises(TypeError):
            palette = [1, 2, 3]
            pygame.display.set_palette(palette)

    skip_list = ["dummy", "android"]
    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") in skip_list,
        'requires the SDL_VIDEODRIVER to be non dummy'
    )
    def test_toggle_fullscreen(self):
        """Test for toggle fullscreen"""

        #try to toggle fullscreen with no active display
        #this should result in an error
        pygame.display.quit()
        with self.assertRaises(pygame.error):
            pygame.display.toggle_fullscreen()

        pygame.display.init()
        width_height = (640,480)
        test_surf = pygame.display.set_mode(width_height)

        #try to toggle fullscreen
        try:
            pygame.display.toggle_fullscreen()

        except pygame.error:
            self.fail()

        else:
            #if toggle success, the width/height should be a
            #value found in list_modes
            if pygame.display.toggle_fullscreen() == 1:
                boolean = (test_surf.get_width(), test_surf.get_height()) \
                in pygame.display.list_modes(depth=0, flags=pygame.FULLSCREEN, display=0)

                self.assertEqual(boolean, True)

            #if not original width/height should be preserved
            else:
                self.assertEqual((test_surf.get_width(), test_surf.get_height()), \
                width_height)

class DisplayInteractiveTest(unittest.TestCase):

    __tags__ = ["interactive"]

    def test_set_icon_interactive(self):

        # on my computer, the window was being created off-screen
        # thus i set the window position manually
        os.environ['SDL_VIDEO_WINDOW_POS'] = '100,250'

        test_icon = pygame.Surface((32, 32))
        test_icon.fill((255,0,0))

        pygame.display.set_icon(test_icon)
        screen = pygame.display.set_mode((400,100))
        pygame.display.set_caption(
        "Is the icon to the left a red square?"
        )

        response = question("Is the display icon red square?")

        self.assertTrue(response)

        pygame.display.quit()


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
