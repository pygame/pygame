import unittest
import pygame
import time

Clock = pygame.time.Clock


class ClockTypeTest(unittest.TestCase):
    def test_construction(self):
        """Ensure a Clock object can be created"""
        c = Clock()

        self.assertTrue(c, "Clock cannot be constructed")

    def test_get_fps(self):
        """ test_get_fps tests pygame.time.get_fps() """
        # Initialization check, first call should return 0 fps
        c = Clock()
        self.assertEqual(c.get_fps(), 0)
        # Type check get_fps should return float
        self.assertTrue(type(c.get_fps()) == float)
        # Allowable margin of error in fps
        delta = 4
        # Test fps correctness for 100, 60 and 30 fps
        self._fps_test(c, 100, delta)
        self._fps_test(c, 60, delta)
        self._fps_test(c, 30, delta)

    def _fps_test(self, clock, fps, delta):
        """ticks fps times each second, hence get_fps() should return fps"""
        delay_per_frame = 1.0/fps
        for f in range(fps):  # For one second tick and sleep
            clock.tick()
            time.sleep(delay_per_frame)
        # We should get around fps (+- delta)
        self.assertAlmostEqual(clock.get_fps(), fps, delta=delta)

    def todo_test_get_rawtime(self):

        # __doc__ (as of 2008-08-02) for pygame.time.Clock.get_rawtime:

        # Clock.get_rawtime(): return milliseconds
        # actual time used in the previous tick
        #
        # Similar to Clock.get_time(), but this does not include any time used
        # while Clock.tick() was delaying to limit the framerate.
        #

        self.fail()

    def test_get_time(self):
        #Testing parameters
        delay = 0.1 #seconds
        delay_miliseconds = delay*(10**3)
        iterations = 10
        delta = 50 #milliseconds

        #Testing Clock Initialization
        c = Clock()
        self.assertEqual(c.get_time(), 0)

        #Testing within delay parameter range
        for i in range(iterations):
            time.sleep(delay)
            c.tick()
            c1 = c.get_time()
            self.assertAlmostEqual(delay_miliseconds, c1, delta=delta)

        #Comparing get_time() results with the 'time' module
        for i in range(iterations):
            t0 = time.time()
            time.sleep(delay)
            c.tick()
            t1 = time.time()
            c1 = c.get_time() #elapsed time in milliseconds
            d0 = (t1-t0)*(10**3) #'time' module elapsed time converted to milliseconds
            self.assertAlmostEqual(d0, c1, delta=delta)

    def todo_test_tick(self):

        # __doc__ (as of 2008-08-02) for pygame.time.Clock.tick:

        # Clock.tick(framerate=0): return milliseconds
        # control timer events
        # update the clock
        #
        # This method should be called once per frame. It will compute how
        # many milliseconds have passed since the previous call.
        #
        # If you pass the optional framerate argument the function will delay
        # to keep the game running slower than the given ticks per second.
        # This can be used to help limit the runtime speed of a game. By
        # calling Clock.tick(40) once per frame, the program will never run at
        # more than 40 frames per second.
        #
        # Note that this function uses SDL_Delay function which is not
        # accurate on every platform, but does not use much cpu.  Use
        # tick_busy_loop if you want an accurate timer, and don't mind chewing
        # cpu.
        #

        self.fail()

        # collection = []
        # c = Clock()
        #
        # c.tick()
        # for i in range(100):
        #     time.sleep(0.005)
        #     collection.append(c.tick())
        #
        # for outlier in [min(collection), max(collection)]:
        #     if outlier != 5: collection.remove(outlier)
        #
        # self.assertEqual(sum(collection) / len(collection), 5)

    def todo_test_tick_busy_loop(self):

        # __doc__ (as of 2008-08-02) for pygame.time.Clock.tick_busy_loop:

        # Clock.tick_busy_loop(framerate=0): return milliseconds
        # control timer events
        # update the clock
        #
        # This method should be called once per frame. It will compute how
        # many milliseconds have passed since the previous call.
        #
        # If you pass the optional framerate argument the function will delay
        # to keep the game running slower than the given ticks per second.
        # This can be used to help limit the runtime speed of a game. By
        # calling Clock.tick(40) once per frame, the program will never run at
        # more than 40 frames per second.
        #
        # Note that this function uses pygame.time.delay, which uses lots of
        # cpu in a busy loop to make sure that timing is more accurate.
        #
        # New in pygame 1.8.0.

        self.fail()

class TimeModuleTest(unittest.TestCase):
    def test_delay(self):
        """Tests time.delay() function."""
        millis = 50  # millisecond to wait on each iteration
        iterations = 20  # number of iterations
        delta = 50  # Represents acceptable margin of error for wait in ms
        # Call checking function
        self._wait_delay_check(pygame.time.delay, millis, iterations, delta)
        # After timing behaviour, check argument type exceptions
        self._type_error_checks(pygame.time.delay)

    def test_get_ticks(self):
        """Tests time.get_ticks()"""
        """
         Iterates and delays for arbitrary amount of time for each iteration,
         check get_ticks to equal correct gap time
        """
        iterations = 20
        millis = 50
        delta = 15  # Acceptable margin of error in ms
        # Assert return type to be int
        self.assertTrue(type(pygame.time.get_ticks()) == int)
        for i in range(iterations):
            curr_ticks = pygame.time.get_ticks()  # Save current tick count
            curr_time = time.time()  # Save current time
            pygame.time.delay(millis)  # Delay for millis
            # Time and Ticks difference from start of the iteration
            time_diff = round((time.time() - curr_time)*1000)
            ticks_diff = pygame.time.get_ticks() - curr_ticks
            # Assert almost equality of the ticking time and time difference
            self.assertAlmostEqual(ticks_diff, time_diff, delta=delta)

    def todo_test_set_timer(self):

        # __doc__ (as of 2008-08-02) for pygame.time.set_timer:

        # pygame.time.set_timer(eventid, milliseconds): return None
        # repeatedly create an event on the event queue
        #
        # Set an event type to appear on the event queue every given number of
        # milliseconds. The first event will not appear until the amount of
        # time has passed.
        #
        # Every event type can have a separate timer attached to it. It is
        # best to use the value between pygame.USEREVENT and pygame.NUMEVENTS.
        #
        # To disable the timer for an event, set the milliseconds argument to 0.

        self.fail()

    def test_wait(self):
        """Tests time.wait() function."""
        millis = 100  # millisecond to wait on each iteration
        iterations = 10  # number of iterations
        delta = 50  # Represents acceptable margin of error for wait in ms
        # Call checking function
        self._wait_delay_check(pygame.time.wait, millis, iterations, delta)
        # After timing behaviour, check argument type exceptions
        self._type_error_checks(pygame.time.wait)

    def _wait_delay_check(self, func_to_check, millis, iterations, delta):
        """"
         call func_to_check(millis) "iterations" times and check each time if
         function "waited" for given millisecond (+- delta). At the end, take
         average time for each call (whole_duration/iterations), which should
         be equal to millis (+- delta - acceptable margin of error).
         *Created to avoid code duplication during delay and wait tests
        """
        # take starting time for duration calculation
        start_time = time.time()
        for i in range(iterations):
            wait_time = func_to_check(millis)
            # Check equality of wait_time and millis with margin of error delta
            self.assertAlmostEqual(wait_time, millis, delta=delta)
        stop_time = time.time()
        # Cycle duration in millisecond
        duration = round((stop_time-start_time)*1000)
        # Duration/Iterations should be (almost) equal to predefined millis
        self.assertAlmostEqual(duration/iterations, millis, delta=delta)

    def _type_error_checks(self, func_to_check):
        """Checks 3 TypeError (float, tuple, string) for the func_to_check"""
        """Intended for time.delay and time.wait functions"""
        # Those methods throw no exceptions on negative integers
        self.assertRaises(TypeError, func_to_check, 0.1)  # check float
        self.assertRaises(TypeError, pygame.time.delay, (0, 1))  # check tuple
        self.assertRaises(TypeError, pygame.time.delay, "10")  # check string

###############################################################################

if __name__ == "__main__":
    unittest.main()
