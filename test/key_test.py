import unittest
import pygame
import pygame.key

SDL1 = pygame.get_sdl_version()[0] < 2


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

    def test_import(self):
        "does it import"
        import pygame.key

    def test_get_focused(self):
        focused = pygame.key.get_focused()
        if not SDL1:
            self.assertFalse(focused) #No window to focus
        self.assertIsInstance(focused, int)
        display_sizes = pygame.display.list_modes()
        if display_sizes == -1:
            display_sizes = [(500, 500)]
        if not display_sizes:
            return
        pygame.display.set_mode(size = display_sizes[-1], flags = pygame.FULLSCREEN)
        pygame.event.set_grab(True)
        pygame.event.pump() #Pump event queue to get window focus on macos
        if not pygame.event.get_grab():
            return
        focused = pygame.key.get_focused()
        self.assertIsInstance(focused, int)
        self.assertTrue(focused)

    def test_get_pressed(self):
        states = pygame.key.get_pressed()
        self.assertEqual(states[pygame.K_RIGHT], 0)

    def test_name(self):
        self.assertEqual(pygame.key.name(pygame.K_RETURN), "return")
        self.assertEqual(pygame.key.name(pygame.K_0), "0")
        self.assertEqual(pygame.key.name(pygame.K_SPACE), "space")

    def test_key_code(self):
        if SDL1:
            self.assertRaises(NotImplementedError, pygame.key.key_code,
                              "return")
        else:
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
