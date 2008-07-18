#################################### IMPORTS ###################################

import test_utils
import test.unittest as unittest
from test_utils import test_not_implemented

import pygame, time

Clock = pygame.time.Clock

################################################################################

class ClockTypeTest(unittest.TestCase):
    def test_construction(self):
        c = Clock()
        self.assert_(c, "Clock can be constructed")
    
    def test_get_fps(self):
        self.assert_(test_not_implemented())
        return
        
        # __doc__ (as of 2008-07-03) for pygame.time.Clock.get_fps:

          # Clock.get_fps(): return float
          # compute the clock framerate

        delay_per_frame = 1 / 100.0

        c = Clock()

        for f in range(100):
            c.tick()
            time.sleep(delay_per_frame)

        self.assert_(99.0 < c.get_fps() < 101.0)

    def test_tick(self):
        self.assert_(test_not_implemented())
        return

        # __doc__ (as of 2008-07-03) for pygame.time.Clock.tick:

          # Clock.tick(framerate=0): return milliseconds
          # control timer events
          # update the clock
        
        collection = []
        c = Clock()

        c.tick()
        for i in range(100):
            time.sleep(0.005)
            collection.append(c.tick())

        for outlier in [min(collection), max(collection)]:
            if outlier != 5: collection.remove(outlier)

        self.assert_(sum(collection) / len(collection) == 5)


    def test_get_rawtime(self):
        # __doc__ (as of 2008-07-03) for pygame.time.Clock.get_rawtime:

          # Clock.get_rawtime(): return milliseconds
          # actual time used in the previous tick

        self.assert_(test_not_implemented())

    def test_get_time(self):
        
        # __doc__ (as of 2008-07-03) for pygame.time.Clock.get_time:

          # Clock.get_time(): return milliseconds
          # time used in the previous tick

        self.assert_(test_not_implemented())
        return
        
        c = Clock()
        c.tick()                    #between   here 
        time.sleep(0.02)         
                                                     #get_time()
        c.tick()                    #          here

        time.sleep(0.02)
        
        self.assert_(20 <= c.get_time() <= 30)
        
    def test_tick_busy_loop(self):

        # __doc__ (as of 2008-07-03) for pygame.time.Clock.tick_busy_loop:

          # Clock.tick_busy_loop(framerate=0): return milliseconds
          # control timer events
          # update the clock

        self.assert_(test_not_implemented()) 

class TimeModuleTest(unittest.TestCase):
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
    unittest.main()