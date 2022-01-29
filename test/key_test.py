import os
import sys
import time
import unittest
import pygame
import pygame.key


class KeyModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(cls):
        # This makes sure pygame is always initialized before each test (in
        # case a test calls pygame.quit()).
        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()

    def test_import(self):
        """does it import?"""
        import pygame.key

    # fixme: test_get_focused failing systematically in some linux
    # fixme: test_get_focused failing on SDL 2.0.18 on Windows
    @unittest.skip("flaky test, and broken on 2.0.18 windows")
    def test_get_focused(self):
        # This test fails in SDL2 in some linux
        # This test was skipped in SDL1.
        focused = pygame.key.get_focused()
        self.assertFalse(focused)  # No window to focus
        self.assertIsInstance(focused, int)
        # Dummy video driver never gets keyboard focus.
        if os.environ.get("SDL_VIDEODRIVER") != "dummy":
            # Positive test, fullscreen with events grabbed
            display_sizes = pygame.display.list_modes()
            if display_sizes == -1:
                display_sizes = [(500, 500)]
            pygame.display.set_mode(size=display_sizes[-1], flags=pygame.FULLSCREEN)
            pygame.event.set_grab(True)
            # Pump event queue to get window focus on macos
            pygame.event.pump()
            focused = pygame.key.get_focused()
            self.assertIsInstance(focused, int)
            self.assertTrue(focused)
            # Now test negative, iconify takes away focus
            pygame.event.clear()
            # TODO: iconify test fails in windows
            if os.name != "nt":
                pygame.display.iconify()
                # Apparent need to pump event queue in order to make sure iconify
                # happens. See display_test.py's test_get_active_iconify
                for _ in range(50):
                    time.sleep(0.01)
                    pygame.event.pump()
                self.assertFalse(pygame.key.get_focused())
                # Test if focus is returned when iconify is gone
                pygame.display.set_mode(size=display_sizes[-1], flags=pygame.FULLSCREEN)
                for i in range(50):
                    time.sleep(0.01)
                    pygame.event.pump()
                self.assertTrue(pygame.key.get_focused())
        # Test if a quit display raises an error:
        pygame.display.quit()
        with self.assertRaises(pygame.error) as cm:
            pygame.key.get_focused()

    def test_get_pressed(self):
        states = pygame.key.get_pressed()
        self.assertEqual(states[pygame.K_RIGHT], 0)

    def test_name(self):
        self.assertEqual(pygame.key.name(pygame.K_RETURN), "return")
        self.assertEqual(pygame.key.name(pygame.K_0), "0")
        self.assertEqual(pygame.key.name(pygame.K_SPACE), "space")

    def test_key_code(self):
        self.assertEqual(pygame.key.key_code("return"), pygame.K_RETURN)
        self.assertEqual(pygame.key.key_code("0"), pygame.K_0)
        self.assertEqual(pygame.key.key_code("space"), pygame.K_SPACE)

        self.assertRaises(ValueError, pygame.key.key_code, "fizzbuzz")

    def test_set_and_get_mods(self):
        pygame.key.set_mods(pygame.KMOD_CTRL)
        self.assertEqual(pygame.key.get_mods(), pygame.KMOD_CTRL)

        pygame.key.set_mods(pygame.KMOD_ALT)
        self.assertEqual(pygame.key.get_mods(), pygame.KMOD_ALT)
        pygame.key.set_mods(pygame.KMOD_CTRL | pygame.KMOD_ALT)
        self.assertEqual(pygame.key.get_mods(), pygame.KMOD_CTRL | pygame.KMOD_ALT)

    def test_set_and_get_repeat(self):
        self.assertEqual(pygame.key.get_repeat(), (0, 0))

        pygame.key.set_repeat(10, 15)
        self.assertEqual(pygame.key.get_repeat(), (10, 15))

        pygame.key.set_repeat()
        self.assertEqual(pygame.key.get_repeat(), (0, 0))


if __name__ == "__main__":
    unittest.main()
