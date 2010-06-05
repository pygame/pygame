import unittest
import pygame2
import pygame2.sdl.keyboard as keyboard
import pygame2.sdl.video as video
import pygame2.sdl.constants as constants

class SDLKeyboardTest (unittest.TestCase):

    def setUp (self):
        video.init ()

    def tearDown (self):
        video.quit ()

    def test_pygame2_sdl_keyboard_enable_repeat(self):

        # __doc__ (as of 2009-05-13) for pygame2.sdl.keyboard.enable_repeat:

        # enable_repeat (delay, interval) -> None
        # 
        # Enables or disables the keyboard repeat rate.
        # 
        # delay specifies how long a key must be pressed before the
        # repeat begins. interval is the speed with which it
        # repeats. delay and interval are expressed as
        # milliseconds. Thus, after the initial delay has passed,
        # repeated KEYDOWN events are sent through the event queue,
        # using the specified interval.  Setting delay to 0 will disable
        # repeating completely.
        # 
        # Setting delay to 0 will disable repeating completely.
        self.assert_ (keyboard.enable_repeat (0, 0) == None)
        self.assert_ (keyboard.enable_repeat (1, 1) == None)
        self.assert_ (keyboard.enable_repeat (900, 1000) == None)
        self.assertRaises (ValueError, keyboard.enable_repeat, -1, -1)
        self.assertRaises (ValueError, keyboard.enable_repeat,  1, -1)
        self.assertRaises (ValueError, keyboard.enable_repeat, -1,  1)

    def test_pygame2_sdl_keyboard_enable_unicode(self):

        # __doc__ (as of 2009-05-13) for pygame2.sdl.keyboard.enable_unicode:

        # enable_unicode ([enable]) -> bool
        # 
        # Enables or disables unicode input handling.
        # 
        # Enables or disables unicode input handling. If the argument is
        # omitted, the function will return the current unicode handling
        # state.  By default unicode handling is enabled and for
        # keyboard events, the *unicode* member of the event will be
        # filled with the corresponding unicode character.
        self.assertEqual (keyboard.enable_unicode (), True)
        self.assertEqual (keyboard.enable_unicode (True), True)
        self.assertEqual (keyboard.enable_unicode (False), True)
        self.assertEqual (keyboard.enable_unicode (True), False)
        self.assertEqual (keyboard.enable_unicode (True), True)
        self.assertEqual (keyboard.enable_unicode (), True)
        self.assertEqual (keyboard.enable_unicode (False), True)
        self.assertEqual (keyboard.enable_unicode (False), False)
        self.assertEqual (keyboard.enable_unicode (), False)
        
    def test_pygame2_sdl_keyboard_get_key_name(self):

        # __doc__ (as of 2009-05-13) for pygame2.sdl.keyboard.get_key_name:

        # get_key_name (key) -> str
        # 
        # Get the descriptive name for a key constant.
        self.assertEqual (keyboard.get_key_name (constants.K_a), 'a')
        self.assertEqual (keyboard.get_key_name (constants.K_b), 'b')
        self.assertEqual (keyboard.get_key_name (constants.K_q), 'q')
        self.assertEqual (keyboard.get_key_name (constants.K_LEFT), 'left')
        self.assertEqual (keyboard.get_key_name (constants.K_PAGEUP), 'page up')
        self.assertEqual (keyboard.get_key_name (constants.K_KP4), '[4]')
        self.assertEqual (keyboard.get_key_name (constants.K_4), '4')

    def test_pygame2_sdl_keyboard_get_mod_state(self):

        # __doc__ (as of 2009-05-13) for pygame2.sdl.keyboard.get_mod_state:

        # get_mod_state () -> state
        # 
        # Returns the current state of the modifier keys (CTRL, ALT, etc.).
        # 
        # Returns a single integer representing a bitmask of all the
        # modifier keys being held. Using bitwise operators you can test
        # if specific shift keys are pressed, the state of the capslock
        # button, and more.  The bitmask will consist of the various
        # KMOD_* flags as specified in the constants.
        self.assertEqual (keyboard.get_mod_state (), 0)
        kstate = constants.KMOD_LALT|constants.KMOD_NUM
        keyboard.set_mod_state (kstate)
        self.assertEqual (keyboard.get_mod_state (), kstate)
        keyboard.set_mod_state (constants.KMOD_CAPS)
        self.assertEqual (keyboard.get_mod_state (), constants.KMOD_CAPS)
        keyboard.set_mod_state (kstate)
        self.assertEqual (keyboard.get_mod_state (), kstate)

    def test_pygame2_sdl_keyboard_get_repeat(self):

        # __doc__ (as of 2009-05-13) for pygame2.sdl.keyboard.get_repeat:

        # get_repeat () -> delay, interval
        # 
        # Gets the currently set keyboard repeat delay and interval.
        keyboard.enable_repeat (0, 0)
        self.assertEqual (keyboard.get_repeat (), (0, 0))
        keyboard.enable_repeat (10, 10)
        self.assertEqual (keyboard.get_repeat (), (10, 10))
        keyboard.enable_repeat (5, 2)
        self.assertEqual (keyboard.get_repeat (), (5, 2))
        keyboard.enable_repeat (0, 5)
        self.assertEqual (keyboard.get_repeat (), (0, 5))
        keyboard.enable_repeat (7, 0)
        self.assertEqual (keyboard.get_repeat (), (7, 0))

    def test_pygame2_sdl_keyboard_get_state(self):

        # __doc__ (as of 2009-05-13) for pygame2.sdl.keyboard.get_state:

        # get_state () -> dict
        # 
        # Gets the current keyboard state.
        # 
        # Gets a dictionary with the current keyboard state. The keys of
        # the dictionary are the key constants, the boolean values of
        # the dictionary indicate, whether a certain key is pressed or
        # not.
        self.assert_ (type (keyboard.get_state ()) == dict)
        self.assert_ (constants.K_a in keyboard.get_state ().keys ())
        self.assert_ (constants.K_b in keyboard.get_state ().keys ())
        self.assert_ (constants.K_q in keyboard.get_state ().keys ())
        self.assert_ (constants.K_KP4 in keyboard.get_state ().keys ())

    def test_pygame2_sdl_keyboard_set_mod_state(self):

        # __doc__ (as of 2009-05-13) for pygame2.sdl.keyboard.set_mod_state:

        # set_mod_state (mod) -> None
        # 
        # Sets the current modifier key state.
        # 
        # Sets the current modifier key state. mod has to be a bitwise OR'd
        # combination of the KMOD_* flags as they are specified in the
        # constants.
        self.assertEqual (keyboard.get_mod_state (), 0)
        kstate = constants.KMOD_LALT|constants.KMOD_NUM
        keyboard.set_mod_state (kstate)
        self.assertEqual (keyboard.get_mod_state (), kstate)
        keyboard.set_mod_state (constants.KMOD_CAPS)
        self.assertEqual (keyboard.get_mod_state (), constants.KMOD_CAPS)
        keyboard.set_mod_state (kstate)
        self.assertEqual (keyboard.get_mod_state (), kstate)

if __name__ == "__main__":
    unittest.main ()
