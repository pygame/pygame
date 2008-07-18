import test_utils
import test.unittest as unittest
import os

from test_utils import test_not_implemented

import pygame

class KeyModuleTest(unittest.TestCase):
    def test_import(self):
        'does it import'
        import pygame.key

    def test_get_focused(self):
        # __doc__ (as of 2008-06-25) for pygame.key.get_focused:

          # pygame.key.get_focused(): return bool
          # true if the display is receiving keyboard input from the system
        
        self.assert_(test_not_implemented()) 

    def test_get_mods(self):

        # __doc__ (as of 2008-06-25) for pygame.key.get_mods:

          # pygame.key.get_mods(): return int
          # determine which modifier keys are being held

        self.assert_(test_not_implemented()) 

    def test_get_pressed(self):

        # __doc__ (as of 2008-06-25) for pygame.key.get_pressed:

          # pygame.key.get_pressed(): return bools
          # get the state of all keyboard buttons

        self.assert_(test_not_implemented()) 

    def test_name(self):

        # __doc__ (as of 2008-06-25) for pygame.key.name:

          # pygame.key.name(key): return string
          # get the name of a key identifier

        self.assert_(test_not_implemented()) 

    def test_set_mods(self):

        # __doc__ (as of 2008-06-25) for pygame.key.set_mods:

          # pygame.key.set_mods(int): return None
          # temporarily set which modifier keys are pressed

        self.assert_(test_not_implemented()) 

    def test_set_repeat(self):

        # __doc__ (as of 2008-06-25) for pygame.key.set_repeat:

          # pygame.key.set_repeat(): return None
          # pygame.key.set_repeat(delay, interval): return None
          # control how held keys are repeated

        self.assert_(test_not_implemented()) 


if __name__ == '__main__':
    unittest.main()