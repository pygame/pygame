import unittest
import pygame
import time

Clock = pygame._time.Clock

SDL1 = pygame.get_sdl_version()[0] < 2

# Because timing based tests are twitchy, just test stuff again if they fail,
# but only upto NUM_TRIES
NUM_TRIES = 3


# here, delta represents allowable error ratio
def _is_almost_equal(a, b, delta):
    return b * (1 - delta) <= a <= b * (1 + delta)


class ClockTypeTest(unittest.TestCase):
    __tags__ = ['newtiming']

    def test_construction(self):
        """Ensure a Clock object can be created"""
        c = Clock()

        self.assertTrue(c, "Clock cannot be constructed")

    def test_get_fps(self):
        """ test_get_fps tests pygame._time.get_fps() """
        # Initialization check, first call should return 0 fps
        c = Clock()
        self.assertEqual(c.get_fps(), 0)
        # Type check get_fps should return float
        self.assertTrue(isinstance(c.get_fps(), float))

        # Test fps correctness for a few fps
        for i in [24, 30, 60, 100, 120]:
            self._fps_test(c, i)

    def _fps_test(self, clock, fps, reccnt=0):
        """ticks fps times each second, hence get_fps() should return fps"""
        if reccnt == NUM_TRIES:
            self.fail("get_fps does not work")

        delta = 0.2
        delay_per_frame = 1.0 / fps
        for f in range(fps // 4):  # For 250ms tick and sleep
            clock.tick()
            time.sleep(delay_per_frame)

        # We should get around fps (+- fps*delta -- delta % of fps)
        if not _is_almost_equal(clock.get_fps(), fps, delta):
            # recurse on failure
            self._fps_test(clock, fps, reccnt + 1)

    def test_get_rawtime(self):
        iterations = 5
        delay = 0.02
        delay_milliseconds = delay * 1000
        framerate_limit = 20
        delta = 0.1
        errcnt = 0

        # Testing Clock Initialization
        c = Clock()
        self.assertEqual(c.get_rawtime(), 0)

        # Testing Raw Time with Frame Delay
        for f in range(iterations):
            time.sleep(delay)
            c.tick(framerate_limit)
            c1 = c.get_rawtime()
            if not _is_almost_equal(delay_milliseconds, c1, delta):
                errcnt += 1

        if errcnt >= NUM_TRIES:
            self.fail("error while doing get_rawtime()")

        errcnt = 0

        # Testing get_rawtime() = get_time()
        for f in range(iterations):
            time.sleep(delay)
            c.tick()
            c1 = c.get_rawtime()
            c2 = c.get_time()

            self.assertEqual(c1, c2)
            if not _is_almost_equal(delay_milliseconds, c1, delta):
                errcnt += 1

        if errcnt >= NUM_TRIES:
            self.fail("error while doing get_rawtime()")

    def test_get_time(self):
        # Testing parameters
        delay = 0.1  # seconds
        delay_milliseconds = delay * 1000
        iterations = 5
        delta = 0.1
        errcnt = 0

        # Testing Clock Initialization
        c = Clock()
        self.assertEqual(c.get_time(), 0)

        # Testing within delay parameter range
        for i in range(iterations):
            time.sleep(delay)
            c.tick()
            c1 = c.get_time()
            if not _is_almost_equal(delay_milliseconds, c1, delta):
                errcnt += 1

        if errcnt >= NUM_TRIES:
            self.fail("error while doing get_time()")

        errcnt = 0

        # Comparing get_time() results with the 'time' module
        for i in range(iterations):
            t0 = time.time()
            time.sleep(delay)
            c.tick()
            t1 = time.time()
            c1 = c.get_time()  # elapsed time in milliseconds
            # 'time' module elapsed time converted to millis
            d0 = (t1 - t0) * 1000
            if not _is_almost_equal(d0, c1, delta):
                errcnt += 1

        if errcnt >= NUM_TRIES:
            self.fail("error while doing get_time()")

    def test_tick(self):
        """Tests time.Clock.tick()"""
        """
        Loops with a set delay a few times then checks what tick reports to
        verify its accuracy. Then calls tick with a desired frame-rate and
        verifies it is not faster than the desired frame-rate nor is it taking
        a dramatically long time to complete
        """

        # Adjust this value to increase the acceptable sleep jitter
        epsilon = 2
        # Adjust this value to increase the acceptable locked frame-rate jitter
        epsilon2 = 0.5
        # adjust this value to increase the acceptable frame-rate margin
        epsilon3 = 25
        testing_framerate = 60
        milliseconds = 5.0

        collection = []
        c = Clock()

        # verify time.Clock.tick() will measure the time correctly
        c.tick()
        for i in range(50):
            time.sleep(milliseconds / 1000)  # convert to seconds
            collection.append(c.tick())

        # removes the first highest and lowest value
        # remove outliers more than once
        for _ in range(3):
            for outlier in [min(collection), max(collection)]:
                if outlier != milliseconds:
                    collection.remove(outlier)

        average_time = sum(collection) / len(collection)

        # assert the deviation from the intended frame-rate is within the
        # acceptable amount (the delay is not taking a dramatically long time)
        self.assertAlmostEqual(average_time, milliseconds, delta=epsilon)

        # verify tick will control the frame-rate

        c = Clock()
        collection = []

        start = time.time()

        for i in range(testing_framerate // 2):
            collection.append(c.tick(testing_framerate))

        # remove the highest and lowest outliers, do that 3 times
        for _ in range(3):
            for outlier in [min(collection), max(collection)]:
                if outlier != round(1000 / testing_framerate):
                    collection.remove(outlier)

        end = time.time()

        # Since calling tick with a desired fps will prevent the program from
        # running at greater than the given fps, 100 iterations at 100 fps
        # should last for about half second
        self.assertAlmostEqual(end - start, 0.5, delta=epsilon2)

        average_tick_time = sum(collection) / len(collection)
        self.assertAlmostEqual(1000 / average_tick_time,
                               testing_framerate, delta=epsilon3)

        # now call _tick_test(), which are essentially more rigourous tests
        if not SDL1:
            self._tick_test()

    def _tick_test(self, reccnt=0):
        if reccnt >= NUM_TRIES:
            self.fail("clock.tick() failed")

        second_len = 1000.0
        delta = 0.3
        fps = 40

        c = Clock()

        if not _is_almost_equal(c.tick(fps), second_len / fps, delta):
            self._tick_test(reccnt + 1)

        # incur delay between ticks that's faster than fps
        pygame._time.wait(10)
        if not _is_almost_equal(c.tick(fps), second_len / fps, delta):
            self._tick_test(reccnt + 1)

        # incur delay between ticks that's slower than fps
        pygame._time.wait(200)
        # the function must return a value close to 200
        if not _is_almost_equal(c.tick(fps), 200, delta):
            self._tick_test(reccnt + 1)

        # Test a wide range of FPS with all kinds of values
        for fps in [500, 1, 35, 600, 32.75]:
            c = Clock()
            if not _is_almost_equal(c.tick(fps), second_len / fps, delta):
                self._tick_test(reccnt + 1)

        zero_fps = 0
        self.assertAlmostEqual(c.tick(zero_fps), 0, delta=0.5)
        negative_fps = -1
        self.assertAlmostEqual(c.tick(negative_fps), 0, delta=0.5)


class TimeModuleTest(unittest.TestCase):
    __tags__ = ['newtiming']

    def test_get_ticks(self):
        """Tests time.get_ticks()"""
        """
         Iterates and delays for arbitrary amount of time for each iteration,
         check get_ticks to equal correct gap time
        """
        iterations = 20
        millis = 50
        delta = 0.1
        errcnt = 0
        # Assert return type to be int
        self.assertTrue(isinstance(pygame._time.get_ticks(), int))
        for i in range(iterations):
            curr_ticks = pygame._time.get_ticks()  # Save current tick count
            curr_time = time.time()  # Save current time
            pygame._time.wait(millis)  # Delay for millis

            # Time and Ticks difference from start of the iteration
            time_diff = (time.time() - curr_time) * 1000
            ticks_diff = pygame._time.get_ticks() - curr_ticks

            # Assert almost equality of the ticking time and time difference
            if not _is_almost_equal(ticks_diff, time_diff, delta):
                errcnt += 1

        if errcnt >= NUM_TRIES:
            self.fail("get_ticks() failed")

    def test_set_timer(self):
        """Tests time.set_timer()"""
        """
        Tests if a timer will post the correct amount of eventid events in
        the specified delay. Test is posting event objects work.
        Also tests if setting milliseconds to 0 stops the timer and if
        the loops arguments work.
        """
        pygame.init()
        TIMER_EVENT_TYPE = pygame.event.custom_type()
        timer_event = pygame.event.Event(TIMER_EVENT_TYPE)
        delta = 50
        timer_delay = 100
        test_number = 8  # Number of events to read for the test
        events = 0  # Events read

        pygame.event.clear()
        pygame.time.set_timer(TIMER_EVENT_TYPE, timer_delay)

        # Test that 'test_number' events are posted in the right amount of time
        t1 = pygame.time.get_ticks()
        max_test_time = t1 + timer_delay * test_number + delta
        while events < test_number:
            for event in pygame.event.get():
                if event == timer_event:
                    events += 1

            # The test takes too much time
            if pygame.time.get_ticks() > max_test_time:
                break
        pygame.time.set_timer(TIMER_EVENT_TYPE, 0)
        t2 = pygame.time.get_ticks()
        # Is the number ef events and the timing right?
        self.assertEqual(events, test_number)
        self.assertAlmostEqual(timer_delay * test_number, t2-t1, delta=delta)

        # Test that the timer stopped when set with 0ms delay.
        pygame.time.delay(200)
        self.assertNotIn(timer_event, pygame.event.get())

        # Test that the loops=True works
        pygame.time.set_timer(TIMER_EVENT_TYPE, 10, True)
        pygame.time.delay(40)
        self.assertEqual(pygame.event.get().count(timer_event), 1)

        # Test a variety of event objects, test loops argument
        events_to_test = [
            pygame.event.Event(TIMER_EVENT_TYPE),
            pygame.event.Event(TIMER_EVENT_TYPE, foo="9gwz5", baz=12,
                               lol=[124, (34, "")]),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, unicode="a")
        ]
        repeat = 3
        millis = 50
        for e in events_to_test:
            pygame.time.set_timer(e, millis, loops=repeat)
            pygame.time.delay(2 * millis * repeat)
            self.assertEqual(pygame.event.get().count(e), repeat)
        pygame.quit()

    def test_wait(self):
        """Tests time.wait() function."""
        millis = 100  # millisecond to wait on each iteration
        iterations = 10  # number of iterations
        delta = 0.1
        errcnt = 0
        # Call checking function
        # take starting time for duration calculation
        start_time = time.time()
        for i in range(iterations):
            wait_time = pygame._time.wait(millis)
            # Check equality of wait_time and millis with margin of error delta
            if not _is_almost_equal(wait_time, millis, delta):
                errcnt += 1
        stop_time = time.time()

        if errcnt >= NUM_TRIES:
            self.fail("wait() failed")

        # Cycle duration in millisecond
        duration = (stop_time - start_time) * 1000
        # Duration/Iterations should be (almost) equal to predefined millis
        self.assertAlmostEqual(
            duration / iterations,
            millis,
            delta=millis * delta)

        # After timing behaviour, check argument type exceptions
        self.assertRaises(TypeError, pygame._time.wait, (0, 1))  # check tuple
        self.assertRaises(TypeError, pygame._time.wait, "10")  # check string


###############################################################################

if __name__ == "__main__":
    unittest.main()