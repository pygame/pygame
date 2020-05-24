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
        pygame.event.get() # Pump event queue to make window get focus on macos.

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

            while not ready_to_test and time_passed <= 1000.0: # Avoid endless loop
                time_passed += clock.tick()
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEMOTION:
                        ready_to_test = True

            self.assertEqual(found_pos, (x, y))


class MouseModuleTest(MouseTests):

    def todo_test_get_cursor(self):
        """Ensures get_cursor works correctly."""
        self.fail()

    def todo_test_set_cursor(self):
        """Ensures set_cursor works correctly."""
        self.fail()

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
            self.assertIsInstance(value, int)

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
