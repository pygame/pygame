#################################### IMPORTS ###################################

import test_utils, unittest
from test_utils import test_not_implemented

################################################################################

class Test(unittest.TestCase):
    pass

    def test_CD(self):

        # __doc__ (as of 2008-06-25) for pygame.cdrom.CD:

          # pygame.cdrom.CD(id): return CD
          # class to manage a cdrom drive

        self.assert_(test_not_implemented()) 

    def test_get_count(self):

        # __doc__ (as of 2008-06-25) for pygame.cdrom.get_count:

          # pygame.cdrom.get_count(): return count
          # number of cd drives on the system

        self.assert_(test_not_implemented()) 

    def test_get_init(self):

        # __doc__ (as of 2008-06-25) for pygame.cdrom.get_init:

          # pygame.cdrom.get_init(): return bool
          # true if the cdrom module is initialized

        self.assert_(test_not_implemented()) 

    def test_init(self):

        # __doc__ (as of 2008-06-25) for pygame.cdrom.init:

          # pygame.cdrom.init(): return None
          # initialize the cdrom module

        self.assert_(test_not_implemented()) 

    def test_quit(self):

        # __doc__ (as of 2008-06-25) for pygame.cdrom.quit:

          # pygame.cdrom.quit(): return None
          # uninitialize the cdrom module

        self.assert_(test_not_implemented()) 


################################################################################

if __name__ == '__main__':
    test_utils.get_fail_incomplete_tests_option()
    unittest.main()
