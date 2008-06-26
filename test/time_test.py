#################################### IMPORTS ###################################

import test_utils, unittest
from test_utils import test_not_implemented

################################################################################

class ClockTypeTest(unittest.TestCase):
    def test_Clock(self):

        # __doc__ (as of 2008-06-25) for pygame.time.Clock:

          # pygame.time.Clock(): return Clock
          # create an object to help track time

        self.assert_(test_not_implemented()) 

class TimeModuleTest(unittest.TestCase):
    def test_Clock(self):
        # __doc__ (as of 2008-06-25) for pygame.time.Clock:

          # pygame.time.Clock(): return Clock
          # create an object to help track time

        self.assert_(test_not_implemented()) 

    def test_delay(self):

        # __doc__ (as of 2008-06-25) for pygame.time.delay:

          # pygame.time.delay(milliseconds): return time
          # pause the program for an amount of time

        self.assert_(test_not_implemented()) 

    def test_get_ticks(self):

        # __doc__ (as of 2008-06-25) for pygame.time.get_ticks:

          # pygame.time.get_ticks(): return milliseconds
          # get the time in milliseconds

        self.assert_(test_not_implemented()) 

    def test_set_timer(self):

        # __doc__ (as of 2008-06-25) for pygame.time.set_timer:

          # pygame.time.set_timer(eventid, milliseconds): return None
          # repeatedly create an event on the event queue

        self.assert_(test_not_implemented()) 

    def test_wait(self):

        # __doc__ (as of 2008-06-25) for pygame.time.wait:

          # pygame.time.wait(milliseconds): return time
          # pause the program for an amount of time

        self.assert_(test_not_implemented()) 

################################################################################

if __name__ == '__main__':
    test_utils.get_fail_incomplete_tests_option()
    unittest.main()