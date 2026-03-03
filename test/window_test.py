import os
import platform
import unittest

import pygame
from pygame import Window
from pygame.version import SDL

pygame.init()


IS_PYPY = "PyPy" == platform.python_implementation()

pygame.display.init()
is_wayland = pygame.display.get_driver() == "wayland"
pygame.display.quit()


class WindowTypeTest(unittest.TestCase):
    DEFAULT_TITLE = "pygame window"

    def setUp(self):
        self.win = Window()

    # Used to check the presence and set-ability of boolean attributes without
    # testing the results of the actions, since the results are not propagated
    # by the NULL videodriver on CI for some things.
    def bool_attr_test(self, attr):
        setattr(self.win, attr, True)
        setattr(self.win, attr, False)
        setattr(self.win, attr, "a truthy string")
        setattr(self.win, attr, "")
        setattr(self.win, attr, 17)
        self.assertIsInstance(getattr(self.win, attr), bool)

    def test_grab_mouse_keyboard(self):
        self.bool_attr_test("grab_mouse")
        self.bool_attr_test("grab_keyboard")

        self.win.grab_mouse = True
        self.assertTrue(self.win.grab_mouse)
        self.win.grab_mouse = False
        self.assertFalse(self.win.grab_mouse)

        if SDL >= (2, 0, 16):
            self.win.grab_keyboard = True
            self.assertTrue(self.win.grab_keyboard)
            self.win.grab_keyboard = False
            self.assertFalse(self.win.grab_keyboard)

    def test_mouse_keyboard_grabbed(self):
        self.assertIsInstance(getattr(self.win, "mouse_grabbed"), bool)
        self.assertIsInstance(getattr(self.win, "keyboard_grabbed"), bool)
        self.assertRaises(
            AttributeError, lambda: setattr(self.win, "mouse_grabbed", False)
        )
        self.assertRaises(
            AttributeError, lambda: setattr(self.win, "keyboard_grabbed", False)
        )

    def test_title(self):
        self.assertEqual(self.win.title, self.DEFAULT_TITLE)

        title = "Hello_World你好世界こにじわ"
        self.win.title = title
        self.assertEqual(self.win.title, title)

    def test_resizable(self):
        self.bool_attr_test("resizable")

    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") == pygame.NULL_VIDEODRIVER,
        "requires the SDL_VIDEODRIVER to be a non-null value",
    )
    def test_resizable_set(self):
        self.win.resizable = True
        self.assertTrue(self.win.resizable)
        self.win.resizable = False
        self.assertFalse(self.win.resizable)

    def test_borderless(self):
        self.bool_attr_test("borderless")

    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") == pygame.NULL_VIDEODRIVER,
        "requires the SDL_VIDEODRIVER to be a non-null value",
    )
    def test_borderless_set(self):
        self.win.borderless = True
        self.assertTrue(self.win.borderless)
        self.win.borderless = False
        self.assertFalse(self.win.borderless)

    @unittest.skipIf(
        SDL < (2, 0, 16),
        "requires SDL 2.0.16+",
    )
    def test_always_on_top(self):
        self.bool_attr_test("always_on_top")

    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") == pygame.NULL_VIDEODRIVER,
        "requires the SDL_VIDEODRIVER to be a non-null value",
    )
    @unittest.skipIf(
        SDL < (2, 0, 16),
        "requires SDL 2.0.16+",
    )
    @unittest.skipIf(is_wayland, "not supported on wayland")
    def test_always_on_top_set(self):
        self.win.always_on_top = True
        self.assertTrue(self.win.always_on_top)
        self.win.always_on_top = False
        self.assertFalse(self.win.always_on_top)

    @unittest.skipIf(
        SDL < (2, 0, 18)
        or os.environ.get("SDL_VIDEODRIVER") == pygame.NULL_VIDEODRIVER,
        "requires SDL 2.0.18+ and SDL_VIDEODRIVER to be a non-null value",
    )
    def test_mouse_rect(self):
        self.win.mouse_rect = None
        self.assertIsNone(self.win.mouse_rect)

        TEST_MOUSE_RECT = (10, 10, 123, 456)
        self.win.mouse_rect = TEST_MOUSE_RECT
        self.assertIsInstance(self.win.mouse_rect, pygame.Rect)
        self.assertTupleEqual(tuple(self.win.mouse_rect), TEST_MOUSE_RECT)

        self.assertRaises(
            TypeError, lambda: setattr(self.win, "mouse_rect", "Incorrect type")
        )

        # clean the status
        self.win.mouse_rect = None

    def test_size(self):
        self.win.size = (1280, 720)
        self.assertTupleEqual(self.win.size, (1280, 720))

        self.assertRaises(ValueError, lambda: setattr(self.win, "size", (-1, 10)))
        self.assertRaises(ValueError, lambda: setattr(self.win, "size", (1, -10)))
        self.assertRaises(ValueError, lambda: setattr(self.win, "size", (-1, -10)))
        self.assertRaises(ValueError, lambda: setattr(self.win, "size", (0, 0)))

        # test set size when init
        # On Windows, if SDL <= 2.0.22, this test fails
        # This bug have been fixed in SDL 2.24.0
        if SDL > (2, 0, 22):
            win = Window(size=(12, 34))
            self.assertTupleEqual((12, 34), win.size)
            win.destroy()

        self.assertRaises(ValueError, lambda: Window(size=(-1, 0)))

        self.win.size = (640, 480)

    @unittest.skipIf(is_wayland, "not supported on wayland")
    def test_position(self):
        new_pos = (self.win.position[0] + 20, self.win.position[1] + 10)
        self.win.position = new_pos
        self.assertTupleEqual(self.win.position, new_pos)

        self.win.position = pygame.WINDOWPOS_CENTERED

        # raise TypeError if input is some random number instead of flag
        self.assertRaises(TypeError, lambda: setattr(self.win, "position", 123))

        # test set position when init
        win = Window(position=new_pos)
        self.assertTupleEqual(new_pos, win.position)
        win.destroy()

        self.assertRaises(TypeError, lambda: Window(position=123))

    def test_maximum_size(self):
        # Use a new window to prevent being influenced by other tests
        self.win = Window()

        self.win.maximum_size = (50, 50)
        self.assertTupleEqual(self.win.maximum_size, (50, 50))

        # size should be limited to the maximum_size
        self.assertTupleEqual(self.win.size, (50, 50))

        self.win.size = (640, 480)
        # size should not larger than maximum_size
        self.assertTupleEqual(self.win.size, (50, 50))

        if SDL.major >= 3:
            # This part should pass in SDL3
            self.win.maximum_size = (0, 0)  # remove the size limit
            self.win.size = (640, 480)
            self.assertTupleEqual(self.win.size, (640, 480))

        self.assertRaises(
            ValueError, lambda: setattr(self.win, "maximum_size", (-1, -1))
        )
        self.assertRaises(
            TypeError, lambda: setattr(self.win, "maximum_size", "placeholder_str")
        )

    def test_minimum_size(self):
        # Use a new window to prevent being influenced by other tests
        self.win = Window()

        self.win.minimum_size = (700, 700)
        self.assertTupleEqual(self.win.minimum_size, (700, 700))

        # size should be limited to the minimum_size
        self.assertTupleEqual(self.win.size, (700, 700))

        self.win.size = (640, 480)
        # size should not small than minimum_size
        self.assertTupleEqual(self.win.size, (700, 700))

        if SDL.major >= 3:
            # This part should pass in SDL3
            self.win.minimum_size = (0, 0)  # remove the size limit
            self.win.size = (640, 480)
            self.assertTupleEqual(self.win.size, (640, 480))

        self.assertRaises(
            ValueError, lambda: setattr(self.win, "minimum_size", (-1, -1))
        )
        self.assertRaises(
            TypeError, lambda: setattr(self.win, "minimum_size", "null_str")
        )

    def test_min_size_interact_with_max_size(self):
        # Use a new window to prevent being influenced by other tests
        self.win = Window()

        self.win.maximum_size = (100, 100)
        # min size should not larger than max size
        self.assertRaises(
            ValueError, lambda: setattr(self.win, "minimum_size", (200, 200))
        )

        self.win.minimum_size = (90, 90)
        # max size should not smaller than min size
        self.assertRaises(
            ValueError, lambda: setattr(self.win, "maximum_size", (50, 50))
        )

        # minimum size should be able to equal to maximum size
        # This test fails in SDL <= 2.0.12
        # have been fixed after SDL 2.0.18
        if SDL >= (2, 0, 18):
            self.win.minimum_size = (60, 60)
            self.win.maximum_size = (60, 60)
            self.assertTupleEqual(self.win.maximum_size, (60, 60))
            self.assertTupleEqual(self.win.minimum_size, (60, 60))

    def test_opacity(self):
        # Setting is not supported at all with SDL_VIDEODRIVER = pygame.NULL_VIDEODRIVER,
        # all we can test here is that the attribute exists and is a
        # float
        self.assertIsInstance(self.win.opacity, float)

    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") == pygame.NULL_VIDEODRIVER,
        "requires the SDL_VIDEODRIVER to be a non-null value",
    )
    @unittest.skipIf(is_wayland, "not supported on wayland")
    def test_opacity_set(self):
        self.win.opacity = 0.5
        self.assertEqual(self.win.opacity, 0.5)
        self.win.opacity = 0
        self.assertEqual(self.win.opacity, 0)
        self.win.opacity = 1
        self.assertEqual(self.win.opacity, 1)
        self.win.opacity = -0.1
        self.assertEqual(self.win.opacity, 0)
        self.win.opacity = 1.1
        self.assertEqual(self.win.opacity, 1)

        self.assertRaises(TypeError, lambda: setattr(self.win, "opacity", "null str"))

    def test_init_flags(self):
        # test no opengl by default
        win = Window()
        self.assertFalse(win.opengl)
        win.destroy()

        # test borderless
        win = Window(borderless=True)
        self.assertTrue(win.borderless)
        win.destroy()

        # test always_on_top
        if SDL >= (2, 0, 16):
            win = Window(always_on_top=True)
            self.assertTrue(win.always_on_top)
            win.destroy()

        # test resizable
        win = Window(resizable=True)
        self.assertTrue(win.resizable)
        win.destroy()

        # test utility
        win = Window(utility=True)
        self.assertTrue(win.utility)
        win.destroy()

        # should raise a TypeError if keyword is random
        self.assertRaises(TypeError, lambda: Window(aaa=True))
        self.assertRaises(TypeError, lambda: Window(aaa=False))

    def test_set_icon(self):
        self.assertRaises(TypeError, lambda: self.win.set_icon(1234))

        test_icon = pygame.Surface((32, 32))
        test_icon.fill((255, 0, 0))

        return_value = self.win.set_icon(test_icon)
        self.assertIsNone(return_value)

    def test_window_object_repr(self):
        win = Window("awa")
        self.assertEqual(repr(win), f"<Window(title='{win.title}', id={win.id})>")

        win.destroy()
        self.assertEqual(repr(win), "<Window(Destroyed)>")

        pygame.display.set_mode((640, 480))
        win = Window.from_display_module()
        self.assertEqual(repr(win), "<Window(From Display)>")

        pygame.display.quit()
        pygame.init()

    def test_from_display_module(self):
        surf = pygame.display.set_mode((640, 480))

        win1 = Window.from_display_module()
        win2 = Window.from_display_module()

        self.assertIs(win1, win2)
        self.assertIs(win1.get_surface(), surf)

        pygame.display.quit()
        pygame.init()

    @unittest.skipIf(IS_PYPY, "for some reason this test is flaky on pypy")
    def test_window_surface(self):
        win = Window(size=(640, 480))
        surf = win.get_surface()

        self.assertIsInstance(surf, pygame.Surface)

        # test auto resize
        self.assertTupleEqual(win.size, surf.get_size())
        win.size = (100, 100)
        self.assertTupleEqual(win.size, surf.get_size())
        win.size = (1280, 720)
        self.assertTupleEqual(win.size, surf.get_size())

        # window surface should be invalid after the window is destroyed
        win.destroy()
        self.assertRaises(pygame.error, lambda: surf.fill((0, 0, 0)))

    @unittest.skipIf(IS_PYPY, "for some reason this test is flaky on pypy")
    def test_window_surface_with_display_module(self):
        # get_surface() should raise an error if the set_mode() is not called.
        pygame.display.set_mode((640, 480))
        win1 = Window.from_display_module()
        pygame.display.quit()
        pygame.init()
        self.assertRaises(pygame.error, lambda: win1.get_surface())

        # the surface returned by get_surface() should be
        # the surface returned by set_mode()
        surf1 = pygame.display.set_mode((640, 480))
        win2 = Window.from_display_module()
        surf2 = win2.get_surface()
        self.assertIs(surf1, surf2)

    def test_window_flip(self):
        win = Window(size=(640, 480))
        surf = win.get_surface()
        surf.fill((255, 0, 0))

        self.assertRaises(TypeError, lambda: win.flip("an argument"))
        self.assertIs(win.flip(), None)
        win.destroy()

        # creates a new window with no surface associated
        win = Window(size=(640, 480))
        self.assertRaisesRegex(
            pygame.error,
            "the Window has no surface associated with it, did you forget to call Window.get_surface()",
            win.flip,
        )
        win.destroy()

    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") == pygame.NULL_VIDEODRIVER,
        "OpenGL requires a non-null SDL_VIDEODRIVER",
    )
    def test_window_opengl(self):
        win1 = Window(opengl=True)
        self.assertTrue(win1.opengl)
        win1.flip()
        win1.destroy()

        win2 = Window(opengl=False)
        self.assertFalse(win2.opengl)
        win2.get_surface()
        win2.flip()
        win2.destroy()

        pygame.display.set_mode((640, 480), pygame.OPENGL)
        win = Window.from_display_module()
        self.assertTrue(win.opengl)
        pygame.display.quit()
        pygame.init()

    @unittest.skipIf(IS_PYPY, "for some reason this test is flaky on pypy")
    def test_window_subclassable(self):
        class WindowSubclass(Window):
            def __init__(self, title="Different title", size=(640, 480), **flags):
                super().__init__(title, size, pygame.WINDOWPOS_CENTERED, **flags)
                self.attribute = 10

        window = WindowSubclass()
        self.assertTrue(issubclass(WindowSubclass, Window))
        self.assertIsInstance(window, WindowSubclass)
        self.assertEqual(window.title, "Different title")
        self.assertEqual(window.attribute, 10)
        window.destroy()

        pygame.display.set_mode((200, 200))
        window = WindowSubclass.from_display_module()
        self.assertIsInstance(window, WindowSubclass)
        self.assertEqual(window.size, (200, 200))

    @unittest.skipIf(
        SDL < (2, 0, 16),
        "requires SDL 2.0.16+",
    )
    def test_window_flash(self):
        window = pygame.Window()

        with self.assertRaises(TypeError):
            window.flash("string")
            window.flash(2.2)
            window.flash([0])

        with self.assertRaises(ValueError):
            window.flash(-1)
            window.flash(3)

        for operation in [
            pygame.FLASH_CANCEL,
            pygame.FLASH_BRIEFLY,
            pygame.FLASH_UNTIL_FOCUSED,
        ]:
            try:
                result = window.flash(operation)
                self.assertIsNone(result)
            except pygame.error:
                pass

    def test_window_focused(self):
        window = pygame.Window()
        self.assertIsInstance(window.focused, bool)

    def tearDown(self):
        self.win.destroy()


if __name__ == "__main__":
    unittest.main()
