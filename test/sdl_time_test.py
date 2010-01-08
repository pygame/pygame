import time
try:
    import pygame2.test.pgunittest as unittest
    from pygame2.test.pgunittest import doprint, interactive
except:
    import pgunittest as unittest
    from pgunittest import doprint, interactive

import pygame2
import pygame2.sdl as sdl
import pygame2.sdl.time as sdltime
import pygame2.sdl.constants as constants

class SDLTimeTest (unittest.TestCase):

    def test_pygame2_sdl_time_add_timer(self):

        # __doc__ (as of 2010-01-06) for pygame2.sdl.time.add_timer:

        # add_timer (interval, callable[, data]) -> CObject
        # 
        # Adds a timer callback to be called periodically.
        # 
        # Adds a timer callback to be called periodically using the specified
        # *interval*. *callable* can be any callable objet, method or function. On
        # invocation, the optional *data* will be passed to the callable.
        # 
        # This will return an CObject that acts as unique id for the timer callback.
        setargs = []
        
        def _timercb (l, arg1, arg2):
            l.append (arg1)
            l.append (arg2)
            return 10
        
        self.assertRaises (pygame2.Error, sdltime.add_timer, _timercb)
        
        sdltime.init ()
        tobj = sdltime.add_timer (10, _timercb, (setargs, "Hello", "World"))
        self.assert_ (tobj != None)
        t1 = t2 = sdltime.get_ticks ()
        while (t2 - t1 < 100):
            sdltime.delay (1)
            t2 = sdltime.get_ticks ()
        self.assert_ (len (setargs) > 10)
        sdltime.quit ()
        
        self.assertRaises (pygame2.Error, sdltime.add_timer, _timercb)

    def test_pygame2_sdl_time_delay(self):

        # __doc__ (as of 2010-01-06) for pygame2.sdl.time.delay:

        # delay (time) -> None
        # 
        # Delays the execution for a specific time.
        # 
        # Delays the program execution for a specific *time*. The *time* is
        # expressed in milliseconds.
        # 
        # NOTE:
        # 
        # This does *not* require init to be called before.
        for i in range (10):
            prev = time.time ()
            sdltime.delay (20)
            last = time.time ()
            delaytime = (last - prev) * 1000
            self.assert_ (10 < delaytime < 50)

    def test_pygame2_sdl_time_get_ticks(self):

        # __doc__ (as of 2010-01-06) for pygame2.sdl.time.get_ticks:

        # get_ticks () -> long
        # 
        # Gets the number of milliseconds since the initialization of
        # the underlying SDL library.
        # 
        # Gets the number of milliseconds since the initialization of
        # the underlying SDL library. The value will wrap if the program
        # runs for more than ~49 days.

        # Can't test this with useful information...
        self.assertRaises (pygame2.Error, sdltime.get_ticks)
        sdltime.init ()
        sdltime.delay (20)
        self.assert_ (sdltime.get_ticks () > 0)
        sdltime.quit ()
        
    def test_pygame2_sdl_time_init(self):

        # __doc__ (as of 2010-01-06) for pygame2.sdl.time.init:

        # init () -> None
        # 
        # Initializes the timer subsystem of the SDL library.
        self.assertEqual (sdltime.init (), None)
        self.assertTrue (sdltime.was_init ())
        self.assertEqual (sdltime.quit (), None)
        self.assertFalse (sdltime.was_init ())

    def test_pygame2_sdl_time_quit(self):

        # __doc__ (as of 2010-01-06) for pygame2.sdl.time.quit:

        # quit () -> None
        # 
        # Shuts down the timer subsystem of the SDL library.
        # 
        # After calling this function, you should not invoke any class,
        # method or function related to the timer subsystem as they are
        # likely to fail or might give unpredictable results.
        self.assertEqual (sdltime.quit (), None)

    def test_pygame2_sdl_time_remove_timer(self):

        # __doc__ (as of 2010-01-06) for pygame2.sdl.time.remove_timer:

        # remove_timer (timerobj) -> None
        # 
        # Removes a previously added timer callback.
        # 
        # Removes a previously added timer callback and throws an exception, if the
        # passed object is not a matching timer object.
        
        def _timercb (flag):
            flag.append (1)
            return 10;
        
        self.assertRaises (pygame2.Error, sdltime.remove_timer, _timercb)
        sdltime.init ()
        self.assertRaises (TypeError, sdltime.remove_timer, None)
        
        flag = []
        tobj = sdltime.add_timer (10, _timercb, (flag,))
        self.assert_ (tobj != None)
        t1 = t2 = sdltime.get_ticks ()
        while (t2 - t1 < 50):
            sdltime.delay (1)
            t2 = sdltime.get_ticks ()
        self.assertTrue (len (flag) != 0)
        sdltime.remove_timer (tobj)
        self.assertRaises (ValueError, sdltime.remove_timer, tobj)
        
        flag = []
        tobj = sdltime.add_timer (10, _timercb, (flag,))
        sdltime.remove_timer (tobj)
        self.assertRaises (ValueError, sdltime.remove_timer, tobj)
        self.assert_ (tobj != None)
        t1 = t2 = sdltime.get_ticks ()
        while (t2 - t1 < 50):
            sdltime.delay (1)
            t2 = sdltime.get_ticks ()
        self.assert_ (len (flag) == 0)
        sdltime.quit ()
        self.assertRaises (pygame2.Error, sdltime.remove_timer, _timercb)

    def todo_test_pygame2_sdl_time_set_timer(self):

        # __doc__ (as of 2010-01-06) for pygame2.sdl.time.set_timer:

        # set_timer (interval, callable) -> None
        # 
        # Sets a single timer callback to be called periodically.
        # 
        # Sets a single timer callback to be called periodically using the specified
        # *interval* in milliseconds. The timer callback can be reset by passing
        # None as *callable* object.

        self.fail() 

    def test_pygame2_sdl_time_was_init(self):

        # __doc__ (as of 2010-01-06) for pygame2.sdl.time.was_init:

        # was_init () -> bool
        # 
        # Returns, whether the timer subsystem of the SDL library is initialized.
        self.assertFalse (sdltime.was_init ())
        self.assertEqual (sdltime.init (), None)
        self.assertTrue (sdltime.was_init ())
        self.assertEqual (sdltime.quit (), None)
        self.assertFalse (sdltime.was_init ())

if __name__ == "__main__":
    unittest.main ()
