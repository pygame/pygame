#################################### IMPORTS ###################################

import test_utils
import test.unittest as unittest
from test_utils import test_not_implemented

################################################################################

class MouseModuleTest(unittest.TestCase):
    def test_get_cursor(self):
        # __doc__ (as of 2008-06-25) for pygame.mouse.get_cursor:

          # pygame.mouse.get_cursor(): return (size, hotspot, xormasks, andmasks)
          # get the image for the system mouse cursor

        self.assert_(test_not_implemented()) 

    def test_get_focused(self):

        # __doc__ (as of 2008-06-25) for pygame.mouse.get_focused:

          # pygame.mouse.get_focused(): return bool
          # check if the display is receiving mouse input

        self.assert_(test_not_implemented()) 

    def test_get_pos(self):

        # __doc__ (as of 2008-06-25) for pygame.mouse.get_pos:

          # pygame.mouse.get_pos(): return (x, y)
          # get the mouse cursor position

        self.assert_(test_not_implemented()) 

    def test_get_pressed(self):

        # __doc__ (as of 2008-06-25) for pygame.mouse.get_pressed:

          # pygame.moouse.get_pressed(): return (button1, button2, button3)
          # get the state of the mouse buttons

        self.assert_(test_not_implemented()) 

    def test_get_rel(self):

        # __doc__ (as of 2008-06-25) for pygame.mouse.get_rel:

          # pygame.mouse.get_rel(): return (x, y)
          # get the amount of mouse movement

        self.assert_(test_not_implemented()) 

    def test_set_cursor(self):

        # __doc__ (as of 2008-06-25) for pygame.mouse.set_cursor:

          # pygame.mouse.set_cursor(size, hotspot, xormasks, andmasks): return None
          # set the image for the system mouse cursor

        self.assert_(test_not_implemented()) 

    def test_set_pos(self):

        # __doc__ (as of 2008-06-25) for pygame.mouse.set_pos:

          # pygame.mouse.set_pos([x, y]): return None
          # set the mouse cursor position

        self.assert_(test_not_implemented()) 

    def test_set_visible(self):

        # __doc__ (as of 2008-06-25) for pygame.mouse.set_visible:

          # pygame.mouse.set_visible(bool): return bool
          # hide or show the mouse cursor

        self.assert_(test_not_implemented()) 


################################################################################

if __name__ == '__main__':
    unittest.main()
