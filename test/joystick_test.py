#################################### IMPORTS ###################################

import test_utils
import test.unittest as unittest
from test_utils import test_not_implemented

################################################################################

class JoystickTypeTest(unittest.TestCase):
    def test_Joystick(self):
        # __doc__ (as of 2008-06-25) for pygame.joystick.Joystick:
    
          # pygame.joystick.Joystick(id): return Joystick
          # create a new Joystick object
    
        self.assert_(test_not_implemented())

class JoytickModuleTest(unittest.TestCase):
    def test_get_count(self):

        # __doc__ (as of 2008-06-25) for pygame.joystick.get_count:

          # pygame.joystick.get_count(): return count
          # number of joysticks on the system

        self.assert_(test_not_implemented()) 

    def test_get_init(self):

        # __doc__ (as of 2008-06-25) for pygame.joystick.get_init:

          # pygame.joystick.get_init(): return bool
          # true if the joystick module is initialized

        self.assert_(test_not_implemented()) 

    def test_init(self):

        # __doc__ (as of 2008-06-25) for pygame.joystick.init:

          # pygame.joystick.init(): return None
          # initialize the joystick module

        self.assert_(test_not_implemented()) 

    def test_quit(self):

        # __doc__ (as of 2008-06-25) for pygame.joystick.quit:

          # pygame.joystick.quit(): return None
          # uninitialize the joystick module

        self.assert_(test_not_implemented()) 


################################################################################

if __name__ == '__main__':
    unittest.main()
