#!/usr/bin/env python

'''Pygame module for monitoring time.

Times in pygame are represented in milliseconds (1/1000 seconds).
Most platforms have a limited time resolution of around 10 milliseconds.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from SDL import *
import pygame.constants

_event_timers = {}

def get_ticks():
    '''Get the time in milliseconds.

    Return the number of millisconds since pygame.init() was called. Before
    pygame is initialized this will always be 0.

    :rtype: int
    '''
    if not SDL_WasInit(SDL_INIT_TIMER):
        return 0
    return SDL_GetTicks()

def wait(time):
    '''Pause the program for an amount of time.

    Will pause for a given number of milliseconds. This function sleeps the
    process to share the processor with other programs. A program that waits
    for even a few milliseconds will consume very little processor time.
    It is slightly less accurate than the pygame.time.delay() function.
     
    :Parameters:
        `time` : int
            Amount of time to wait, in milliseconds.

    :rtype: int
    :return: the actual number of milliseconds used.
    '''
    if not SDL_WasInit(SDL_INIT_TIMER):
        SDL_InitSubSystem(SDL_INIT_TIMER)

    ticks = max(0, time)
    start = SDL_GetTicks()
    SDL_Delay(ticks)
    return SDL_GetTicks() - start

_WORST_CLOCK_ACCURACY = 12
def delay(time):
    '''Pause the program for an amount of time.

    Will pause for a given number of milliseconds. This function will use the
    processor (in addition to sleeping) in order to make the delay more
    accurate than `wait`.
     
    This returns the actual number of milliseconds used.

    :Parameters:
        `time` : int
            Amount of time to wait, in milliseconds.

    :rtype: int
    :return: the actual number of milliseconds used.
    '''
    ticks = max(0, time)
    if not SDL_WasInit(SDL_INIT_TIMER):
        SDL_InitSubSystem(SDL_INIT_TIMER)

    funcstart = SDL_GetTicks()

    delay = ticks - 2 - (ticks % _WORST_CLOCK_ACCURACY)
    if delay >= _WORST_CLOCK_ACCURACY:
        SDL_Delay(delay)

    delay = 1
    while delay > 0:
        delay = ticks - (SDL_GetTicks() - funcstart)

    return SDL_GetTicks() - funcstart

def _timer_callback(interval, param):
    if SDL_WasInit(SDL_INIT_VIDEO):
        event = SDL_Event()
        event.type = param
        SDL_PushEvent(event)
    return interval

def set_timer(event, interval):
    '''Repeatedly create an event on the event queue.

    Set an event type to appear on the event queue every given number of
    milliseconds. The first event will not appear until the amount of time has
    passed.

    Every event type can have a separate timer attached to it. It is best to
    use the value between pygame.USEREVENT and pygame.NUMEVENTS.

    To disable the timer for an event, set the `interval` argument to 0.

    :Parameters:
        `event` : int
            ID of event posted when timer expires.
        `interval` : int
            Interval between events, in milliseconds.

    '''
    if event <= SDL_NOEVENT or event >= SDL_NUMEVENTS:
        raise ValueError, \
              'Event id must be between NOEVENT (%d) and NUMEVENTS (%d)' % \
                (pygame.constants.NOEVENT, pygame.constants.NUMEVENTS)
    
    if event in _event_timers:
        SDL_RemoveTimer(_event_timers[event])

    if interval <= 0:
        return

    if not SDL_WasInit(SDL_INIT_TIMER):
        SDL_InitSubSystem(SDL_INIT_TIMER)

    newtimer = SDL_AddTimer(interval, _timer_callback, event)
    _event_timers[event] = newtimer

class Clock:
    '''An object to help track time.

    The clock also provides several functions to help control a game's
    framerate.
    '''

    def __init__(self):
        '''Create a new Clock object.
        '''
        if not SDL_WasInit(SDL_INIT_TIMER):
            SDL_InitSubSystem(SDL_INIT_TIMER)

        self._fps_tick = 0
        self._last_tick = SDL_GetTicks()
        self._fps = 0.0
        self._fps_count = 0
        self._raw_passed = 0
        self._time_passed = 0

    def __repr__(self):
        return '<Clock(fps=%.2f)>' % self._fps

    def __str__(self):
        return repr(self)

    def _tick(self, framerate, busy_loop):
        if framerate:
            end_time = int(1000 / framerate)
            self._raw_passed = SDL_GetTicks() - self._last_tick
            delay_time = end_time - self._raw_passed

            if busy_loop:
                delay_time = delay(delay_time)
            else:
                delay_time = max(0, delay_time)
                SDL_Delay(delay_time)
        
        nowtime = SDL_GetTicks()
        self._time_passed = nowtime - self._last_tick
        self._fps_count += 1
        self._last_tick = nowtime

        if not framerate:
            self._raw_passed = self._time_passed

        if not self._fps_tick:
            self._fps_count = 0
            self._fps_tick = nowtime
        elif self._fps_count >= 10:
            self._fps = self._fps_count / ((nowtime - self._fps_tick) / 1000.0)
            self._fps_count = 0
            self._fps_tick = nowtime
        return self._time_passed

    def tick(self, framerate=0):
        '''Update the clock.

        This method (or `tick_busy_loop`) should be called once per frame. It
        will compute how many milliseconds have passed since the previous
        call.

        If you pass the optional framerate argument the function will delay to
        keep the game running slower than the given ticks per second.  This
        can be used to help limit the runtime speed of a game. By calling
        Clock.tick(40) once per frame, the program will never run at more than
        40 frames per second.

        Note that this function uses SDL_Delay function which is not accurate
        on every platform, but does not use much cpu.  Use tick_busy_loop
        if you want an accurate timer, and don't mind chewing cpu.

        :Parameters:
            `framerate` : int
                If specified, the maximum framerate to run at.

        :rtype: int
        :return: number of milliseconds since the last tick.
        '''
        return self._tick(framerate, False)

    def tick_busy_loop(self, framerate=0):
        '''Update the clock without yielding the CPU.

        This method (or `tick`) should be called once per frame. It will
        compute how many milliseconds have passed since the previous call.

        If you pass the optional framerate argument the function will delay to
        keep the game running slower than the given ticks per second.  This
        can be used to help limit the runtime speed of a game. By calling
        Clock.tick(40) once per frame, the program will never run at more than
        40 frames per second.

        Note that this function uses `delay`, which uses lots of CPU
        in a busy loop to make sure that timing is more acurate.
        
        :Parameters:
            `framerate` : int
                If specified, the maximum framerate to run at.

        :rtype: int
        :return: number of milliseconds since the last tick.
        '''
        return self._tick(framerate, True)

    def get_time(self):
        '''Time used in the previous tick.

        Returns the number of milliseconds passed between the previous two
        calls to `tick` or `tick_busy_loop`.

        :rtype: int
        '''
        return self._time_passed

    def get_rawtime(self):
        '''Actual time used in the previous tick.

        Similar to `get_time`, but this does not include any time used
        while `tick` or `tick_busy_loop` were delaying to limit the framerate.

        :rtype: int
        '''
        return self._raw_passed

    def get_fps(self):
        '''Compute the clock framerate.

        Compute your game's framerate (in frames per second). It is computed
        by averaging the last few calls to `tick` or `tick_busy_loop`.

        :rtype: float
        '''
        return self._fps
