if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

import unittest


class KeyModuleTest(unittest.TestCase):
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

    def todo_test_get_mods(self):

        # __doc__ (as of 2008-08-02) for pygame.key.get_mods:

          # pygame.key.get_mods(): return int
          # determine which modifier keys are being held
          # 
          # Returns a single integer representing a bitmask of all the modifier
          # keys being held. Using bitwise operators you can test if specific
          # shift keys are pressed, the state of the capslock button, and more.
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

    def todo_test_get_repeat(self):

        # __doc__ (as of 2008-08-02) for pygame.key.get_repeat:

          # pygame.key.get_repeat(): return (delay, interval)
          # see how held keys are repeated
          # 
          # When the keyboard repeat is enabled, keys that are held down will
          # generate multiple pygame.KEYDOWN events. The delay is the number of
          # milliseconds before the first repeated pygame.KEYDOWN will be sent.
          # After that another pygame.KEYDOWN will be sent every interval
          # milliseconds.
          # 
          # When pygame is initialized the key repeat is disabled. 
          # New in pygame 1.8. 

        self.fail() 

    def todo_test_name(self):

        # __doc__ (as of 2008-08-02) for pygame.key.name:

          # pygame.key.name(key): return string
          # get the name of a key identifier
          # 
          # Get the descriptive name of the button from a keyboard button id constant. 

        self.fail() 

    def todo_test_set_mods(self):

        # __doc__ (as of 2008-08-02) for pygame.key.set_mods:

          # pygame.key.set_mods(int): return None
          # temporarily set which modifier keys are pressed
          # 
          # Create a bitmask of the modifier constants you want to impose on your program. 

        self.fail() 

    def todo_test_set_repeat(self):

        # __doc__ (as of 2008-08-02) for pygame.key.set_repeat:

          # pygame.key.set_repeat(): return None
          # pygame.key.set_repeat(delay, interval): return None
          # control how held keys are repeated
          # 
          # When the keyboard repeat is enabled, keys that are held down will
          # generate multiple pygame.KEYDOWN events. The delay is the number of
          # milliseconds before the first repeated pygame.KEYDOWN will be sent.
          # After that another pygame.KEYDOWN will be sent every interval
          # milliseconds. If no arguments are passed the key repeat is disabled.
          # 
          # When pygame is initialized the key repeat is disabled. 

        self.fail() 

if __name__ == '__main__':
    unittest.main()
