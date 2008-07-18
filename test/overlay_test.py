#################################### IMPORTS ###################################

import test_utils
import test.unittest as unittest
from test_utils import test_not_implemented

################################################################################

class OverlayTypeTest(unittest.TestCase):
    def test_display(self):

        # __doc__ (as of 2008-06-25) for pygame.overlay.overlay.display:

          # Overlay.display((y, u, v)): return None
          # Overlay.display(): return None
          # set the overlay pixel data

        self.assert_(test_not_implemented()) 

    def test_get_hardware(self):

        # __doc__ (as of 2008-06-25) for pygame.overlay.overlay.get_hardware:

          # Overlay.get_hardware(rect): return int
          # test if the Overlay is hardware accelerated

        self.assert_(test_not_implemented()) 

    def test_set_location(self):

        # __doc__ (as of 2008-06-25) for pygame.overlay.overlay.set_location:

          # Overlay.set_location(rect): return None
          # control where the overlay is displayed

        self.assert_(test_not_implemented()) 


################################################################################

if __name__ == '__main__':
    unittest.main()
