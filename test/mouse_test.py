import unittest
import os
import platform
import pygame

SDL1 = pygame.get_sdl_version()[0] < 2
DARWIN = "Darwin" in platform.platform()


class MouseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The display needs to be initialized for mouse functions.
        pygame.display.init()

    @classmethod
    def tearDownClass(cls):
        pygame.display.quit()


class MouseModuleInteractiveTest(MouseTests):

    __tags__ = ["interactive"]

    @unittest.skipIf(SDL1 and DARWIN, "Can fails on Mac SDL1, window not focused")
    def test_set_pos(self):
        """ Ensures set_pos works correctly.
            Requires tester to move the mouse to be on the window.
        """
        pygame.display.set_mode((500, 500))
        pygame.event.get()  # Pump event queue to make window get focus on macos.

        if not pygame.mouse.get_focused():
            # The window needs to be focused for the mouse.set_pos to work on macos.
            return
        clock = pygame.time.Clock()

        expected_pos = ((10, 0), (0, 0), (499, 0), (499, 499), (341, 143), (94, 49))

        for x, y in expected_pos:
            pygame.mouse.set_pos(x, y)
            pygame.event.get()
            found_pos = pygame.mouse.get_pos()

            clock.tick()
            time_passed = 0.0
            ready_to_test = False

            while not ready_to_test and time_passed <= 1000.0:  # Avoid endless loop
                time_passed += clock.tick()
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEMOTION:
                        ready_to_test = True

            self.assertEqual(found_pos, (x, y))


class MouseModuleTest(MouseTests):
    def test_get_cursor(self):
        """Ensures get_cursor works correctly."""
        if not SDL1:
            with self.assertRaises(TypeError):
                pygame.mouse.get_cursor()
        else:
            # error should be raised when the display is unintialized
            with self.assertRaises(pygame.error):
                pygame.display.quit()
                pygame.mouse.get_cursor()

            pygame.display.init()

            size = (8, 8)
            hotspot = (0, 0)
            xormask = (0, 96, 120, 126, 112, 96, 0, 0)
            andmask = (224, 240, 254, 255, 254, 240, 96, 0)

            expected_length = 4
            expected_cursor = (size, hotspot, xormask, andmask)

            try:
                cursor = pygame.mouse.get_cursor()

                self.assertIsInstance(cursor, tuple)
                self.assertEqual(len(cursor), expected_length)

                for info in cursor:
                    self.assertIsInstance(info, tuple)

                pygame.mouse.set_cursor(size, hotspot, xormask, andmask)
                self.assertEqual(pygame.mouse.get_cursor(), expected_cursor)

            # SDLError should be raised when the mouse cursor is NULL
            except pygame.error:
                with self.assertRaises(pygame.error):
                    pygame.mouse.get_cursor()

    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER", "") == "dummy",
        "mouse.set_sytem_cursor only available in SDL2",
    )
    def test_set_system_cursor(self):
        """Ensures set_system_cursor works correctly."""

        # Error should be raised when the display is uninitialized
        with self.assertRaises(pygame.error):
            pygame.display.quit()
            pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_HAND)

        pygame.display.init()

        # TypeError raised when PyArg_ParseTuple fails to parse parameters
        with self.assertRaises(TypeError):
            pygame.mouse.set_system_cursor("b")
        with self.assertRaises(TypeError):
            pygame.mouse.set_system_cursor(None)
        with self.assertRaises(TypeError):
            pygame.mouse.set_system_cursor((8, 8), (0, 0))

        # Right type, invalid value
        with self.assertRaises(pygame.error):
            pygame.mouse.set_system_cursor(2000)

        # Working as intended
        self.assertEqual(
            pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_ARROW), None
        )

    @unittest.skipIf(not SDL1, "mouse.get_cursor only available in SDL1")
    def test_set_cursor_sdl1(self):
        """Ensures set_cursor works correctly."""
        size = (8, 8)
        hotspot = (0, 0)
        xormask = (0, 126, 64, 64, 32, 16, 0, 0)
        andmask = (254, 255, 254, 112, 56, 28, 12, 0)
        expected_cursor = (size, hotspot, xormask, andmask)

        # Error should be raised when the display is uninitialized
        with self.assertRaises(pygame.error):
            pygame.display.quit()
            pygame.mouse.set_cursor(size, hotspot, xormask, andmask)

        pygame.display.init()

        # TypeError raised when PyArg_ParseTuple fails to parse parameters
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(("w", "h"), hotspot, xormask, andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, ("0", "0"), xormask, andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, ("x", "y", "z"), xormask, andmask)

        # TypeError raised when either mask is not a sequence
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, hotspot, 12345678, andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, hotspot, xormask, 12345678)

        # TypeError raised when element of mask is not an integer
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, hotspot, "00000000", andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, hotspot, xormask, (2, [0], 4, 0, 0, 8, 0, 1))

        # ValueError raised when width not divisible by 8
        with self.assertRaises(ValueError):
            pygame.mouse.set_cursor((3, 8), hotspot, xormask, andmask)

        # ValueError raised when length of either mask != width * height / 8
        with self.assertRaises(ValueError):
            pygame.mouse.set_cursor((16, 2), hotspot, (128, 64, 32), andmask)
        with self.assertRaises(ValueError):
            pygame.mouse.set_cursor((16, 2), hotspot, xormask, (192, 96, 48, 0, 1))

        # Working as intended
        try:
            self.assertEqual(
                pygame.mouse.set_cursor((16, 1), hotspot, (8, 0), (0, 192)), None
            )
            pygame.mouse.set_cursor(size, hotspot, xormask, andmask)
            self.assertEqual(pygame.mouse.get_cursor(), expected_cursor)
            pygame.mouse.set_cursor(size, hotspot, list(xormask), list(andmask))
            self.assertEqual(pygame.mouse.get_cursor(), expected_cursor)

        # SDLError should be raised when mouse cursor is NULL
        except pygame.error:
            with self.assertRaises(pygame.error):
                pygame.mouse.set_cursor(size, hotspot, xormask, andmask)

    @unittest.skipIf(
        SDL1 or os.environ.get("SDL_VIDEODRIVER", "") == "dummy",
        "Fails on SDL2 with dummy video driver, unable to use mouse.get_cursor",
    )
    def test_set_cursor_sdl2(self):
        """Ensures set_cursor works correctly."""
        size = (8, 8)
        hotspot = (0, 0)
        xormask = (0, 126, 64, 64, 32, 16, 0, 0)
        andmask = (254, 255, 254, 112, 56, 28, 12, 0)

        # Error should be raised when the display is uninitialized
        with self.assertRaises(pygame.error):
            pygame.display.quit()
            pygame.mouse.set_cursor(size, hotspot, xormask, andmask)

        pygame.display.init()

        # TypeError raised when PyArg_ParseTuple fails to parse parameters
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(("w", "h"), hotspot, xormask, andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, ("0", "0"), xormask, andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, ("x", "y", "z"), xormask, andmask)

        # TypeError raised when either mask is not a sequence
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, hotspot, 12345678, andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, hotspot, xormask, 12345678)

        # TypeError raised when element of mask is not an integer
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, hotspot, "00000000", andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, hotspot, xormask, (2, [0], 4, 0, 0, 8, 0, 1))

        # ValueError raised when width not divisible by 8
        with self.assertRaises(ValueError):
            pygame.mouse.set_cursor((3, 8), hotspot, xormask, andmask)

        # ValueError raised when length of either mask != width * height / 8
        with self.assertRaises(ValueError):
            pygame.mouse.set_cursor((16, 2), hotspot, (128, 64, 32), andmask)
        with self.assertRaises(ValueError):
            pygame.mouse.set_cursor((16, 2), hotspot, xormask, (192, 96, 48, 0, 1))

        # Working as intended, no checks for same value as mouse.get_cursor in SDL2
        try:
            self.assertEqual(
                pygame.mouse.set_cursor(size, hotspot, xormask, andmask), None
            )
            self.assertEqual(
                pygame.mouse.set_cursor(size, hotspot, list(xormask), list(andmask)),
                None,
            )

        # SDLError should be raised when mouse cursor is NULL
        except pygame.error:
            with self.assertRaises(pygame.error):
                pygame.mouse.set_cursor(size, hotspot, xormask, andmask)

    def test_get_focused(self):
        """Ensures get_focused returns the correct type."""
        focused = pygame.mouse.get_focused()

        self.assertIsInstance(focused, int)

    def test_get_pressed(self):
        """Ensures get_pressed returns the correct types."""
        expected_length = 3
        buttons_pressed = pygame.mouse.get_pressed()
        self.assertIsInstance(buttons_pressed, tuple)
        self.assertEqual(len(buttons_pressed), expected_length)
        for value in buttons_pressed:
            self.assertIsInstance(value, bool)

        expected_length = 5
        buttons_pressed = pygame.mouse.get_pressed(num_buttons=5)
        self.assertIsInstance(buttons_pressed, tuple)
        self.assertEqual(len(buttons_pressed), expected_length)
        for value in buttons_pressed:
            self.assertIsInstance(value, bool)

        expected_length = 3
        buttons_pressed = pygame.mouse.get_pressed(3)
        self.assertIsInstance(buttons_pressed, tuple)
        self.assertEqual(len(buttons_pressed), expected_length)
        for value in buttons_pressed:
            self.assertIsInstance(value, bool)

        expected_length = 5
        buttons_pressed = pygame.mouse.get_pressed(5)
        self.assertIsInstance(buttons_pressed, tuple)
        self.assertEqual(len(buttons_pressed), expected_length)
        for value in buttons_pressed:
            self.assertIsInstance(value, bool)

        with self.assertRaises(ValueError):
            pygame.mouse.get_pressed(4)

    def test_get_pos(self):
        """Ensures get_pos returns the correct types."""
        expected_length = 2

        pos = pygame.mouse.get_pos()

        self.assertIsInstance(pos, tuple)
        self.assertEqual(len(pos), expected_length)
        for value in pos:
            self.assertIsInstance(value, int)

    def test_set_pos__invalid_pos(self):
        """Ensures set_pos handles invalid positions correctly."""
        for invalid_pos in ((1,), [1, 2, 3], 1, "1", (1, "1"), []):

            with self.assertRaises(TypeError):
                pygame.mouse.set_pos(invalid_pos)

    def test_get_rel(self):
        """Ensures get_rel returns the correct types."""
        expected_length = 2

        rel = pygame.mouse.get_rel()

        self.assertIsInstance(rel, tuple)
        self.assertEqual(len(rel), expected_length)
        for value in rel:
            self.assertIsInstance(value, int)

    def test_get_visible(self):
        """Ensures get_visible works correctly."""
        for expected_value in (False, True):
            pygame.mouse.set_visible(expected_value)

            visible = pygame.mouse.get_visible()

            self.assertEqual(visible, expected_value)

    def test_set_visible(self):
        """Ensures set_visible returns the correct values."""
        # Set to a known state.
        pygame.mouse.set_visible(True)

        for expected_visible in (False, True):
            prev_visible = pygame.mouse.set_visible(expected_visible)

            self.assertEqual(prev_visible, not expected_visible)

    def test_set_visible__invalid_value(self):
        """Ensures set_visible handles invalid positions correctly."""
        for invalid_value in ((1,), [1, 2, 3], 1.1, "1", (1, "1"), []):
            with self.assertRaises(TypeError):
                prev_visible = pygame.mouse.set_visible(invalid_value)


################################################################################

if __name__ == "__main__":
    unittest.main()
