import unittest
import pygame
import pygame.key


class KeyModuleTest(unittest.TestCase):
    def setUp(self):
        pygame.init()

    def tearDown(self):
        pygame.quit()

    def test_import(self):
        'does it import'
        import pygame.key

    def todo_test_get_focused(self):

        # __doc__ (as of 2008-08-02) for pygame.key.get_focused:

          # pygame.key.get_focused(): return bool
          # true if the display is receiving keyboard input from the system
          #
          # This is true when the display window has keyboard focus from the
          # system. If the display needs to ensure it does not lose keyboard
          # focus, it can use pygame.event.set_grab() to grab all input.
          #

        self.fail()

    def todo_test_get_pressed(self):

        # __doc__ (as of 2008-08-02) for pygame.key.get_pressed:

          # pygame.key.get_pressed(): return bools
          # get the state of all keyboard buttons
          #
          # Returns a sequence of boolean values representing the state of every
          # key on the keyboard. Use the key constant values to index the array.
          # A True value means the that button is pressed.
          #
          # Getting the list of pushed buttons with this function is not the
          # proper way to handle text entry from the user. You have no way to
          # know the order of keys pressed, and rapidly pushed keys can be
          # completely unnoticed between two calls to pygame.key.get_pressed().
          # There is also no way to translate these pushed keys into a fully
          # translated character value. See the pygame.KEYDOWN events on the
          # event queue for this functionality.
          #

        self.fail()

    def test_name(self):
        self.assertEqual(pygame.key.name(pygame.K_RETURN), "return")
        self.assertEqual(pygame.key.name(pygame.K_0), "0")

    def test_set_and_get_mods(self):
        self.assertEqual(pygame.key.get_mods(), pygame.KMOD_NONE)

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


if __name__ == '__main__':
    unittest.main()
