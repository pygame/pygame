#!/usr/bin/env python

'''Pygame module for interacting with joystick devices.

The joystick module manages the joystick devices on a computer
(there can be more than one). Joystick devices include
trackballs and video-game-style gamepads, and the module allows
the use of multiple buttons and "hats".
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from SDL import *

import pygame.base

_joysticks = []

def __PYGAMEinit__():
    if not SDL_WasInit(SDL_INIT_JOYSTICK):
        SDL_InitSubSystem(SDL_INIT_JOYSTICK)
        SDL_JoystickEventState(SDL_ENABLE)
        pygame.base.register_quit(_joystick_autoquit)
    return 1

def _joystick_autoquit():
    global _joysticks
    for joystick in _joysticks:
        SDL_JoystickClose(joystick)
    _joysticks = []

    if SDL_WasInit(SDL_INIT_JOYSTICK):
        SDL_JoystickEventState(SDL_DISABLE)  # XXX: Pygame enables here
        SDL_QuitSubSystem(SDL_INIT_JOYSTICK)

def _joystick_init_check():
    if not SDL_WasInit(SDL_INIT_JOYSTICK):
        raise pygame.base.error, 'joystick system not initialized'

def init():
    '''Initialize the joystick module.

    This function is called automatically by pygame.init().

    It initializes the joystick module. This will scan the system for all
    joystick devices. The module must be initialized before any other
    functions will work.

    It is safe to call this function more than once.
    '''
    __PYGAMEinit__()

def quit():
    '''Uninitialize the joystick module.

    Uninitialize the joystick module. After you call this any existing joystick
    objects will no longer work.

    It is safe to call this function more than once.
    '''
    _joystick_autoquit()

def get_init():
    '''Determine if the joystick module is initialized.

    Test if the pygame.joystick.init() function has been called.

    :rtype: bool
    '''
    return SDL_WasInit(SDL_INIT_JOYSTICK) != 0

def get_count():
    '''Number of joysticks on the system.

    Return the number of joystick devices on the system. The
    count will be 0 if there are no joysticks on the system.

    When you create Joystick objects using Joystick(id), you pass an integer
    that must be lower than this count.

    :rtype: int
    '''
    _joystick_init_check()

    return SDL_NumJoysticks()

class Joystick(object):
    '''Joystick representation.
    
    To access most of the Joystick methods, you'll need to init() 
    the Joystick. This is separate from making sure the joystick module is
    initialized. When multiple Joysticks objects are created for the
    same physical joystick device (i.e., they have the same ID number),
    the state and values for those Joystick objects
    will be shared.

    The Joystick object allows you to get information about the types of
    controls on a joystick device. Once the device is initialized the Pygame
    event queue will start receiving events about its input.
     
    You can call the Joystick.get_name() and Joystick.get_id() functions
    without initializing the Joystick object.
    '''

    __slots__ = ['_id', '_device']

    def __init__(self, id):
        '''Create a new Joystick object.

        Create a new joystick to access a physical device. The id argument
        must be a value from 0 to pygame.joystick.get_count()-1.
        '''
        if id < 0 or id >= SDL_NumJoysticks():
            raise pygame.base.error, 'Invalid joystick device number'

        self._id = id

    def init(self):
        '''Initialize the Joystick.

        The Joystick must be initialized to get most of the information about
        the controls. While the Joystick is initialized the Pygame event queue
        will receive events from the Joystick input.

        It is safe to call this more than once.
        '''
        _joystick_init_check()

        if not self._device:
            self._device = SDL_JoystickOpen(self._id)
            _joysticks.append(self._device)


    def quit(self):
        '''Uninitialize the Joystick.

        This will unitialize a Joystick. After this the Pygame event queue
        will no longer receive events from the device.

        It is safe to call this more than once.
        '''
        _joystick_init_check()

        if self._device:
            SDL_JoystickClose(self._device)
            _joysticks.remove(self._device)
            self._device = None

    def get_init(self):
        '''Check if the Joystick is initialized.

        Returns True if the init() method has already been called
        on this Joystick object.
    
        :rtype: bool
        '''
        return self._device is not None

    def _init_check(self):
        _joystick_init_check()
        if not self._device:
            raise pygame.base.error, 'Joystick not initialized'

    def get_id(self):
        '''Get the Joystick ID.

        Returns the integer ID that represents this device. This is the same
        value that was passed to the Joystick() constructor. This method can
        safely be called while the Joystick is not initialized.

        :rtype: int
        '''
        return self._id

    def get_name(self):
        '''Get the Joystick system name.

        Returns the system name for this joystick device. It is unknown what
        name the system will give to the Joystick, but it should be a unique
        name that identifies the device. This method can safely be called
        while the Joystick is not initialized.

        :rtype: str
        '''
        _joystick_init_check()

        return SDL_JoystickName(self._id)

    def get_numaxes(self):
        '''Get the number of axes on a Joystick.

        Returns the number of input axes are on a Joystick. There will usually
        be two for the position. Controls like rudders and throttles are
        treated as additional axes.

        The pygame.JOYAXISMOTION events will be in the range from -1.0 to 1.0.
        A value of 0.0 means the axis is centered. Gamepad devices will
        usually be -1, 0, or 1 with no values in between. Older analog
        joystick axes will not always use the full -1 to 1 range, and the
        centered value will be some area around 0.  Analog joysticks usually
        have a bit of noise in their axis, which will generate a lot of rapid
        small motion events.

        :rtype: int
        '''
        self._init_check()
        return SDL_JoystickNumAxes(self._device)

    def get_axis(self, axis):
        '''Get the current position of an axis.

        Returns the current position of a joystick axis. The value will range
        from -1 to 1 with a value of 0 being centered. You may want to take
        into account some tolerance to handle jitter, and joystick drift may
        keep the joystick from centering at 0 or using the full range of
        position values.

        The axis number must be an integer from zero to get_numaxes()-1.

        :Parameters:
            `axis` : int
                Axis to read.

        :rtype: float
        '''
        self._init_check()
        if axis < 0 or axis >= SDL_JoystickNumAxes(self._device):
            raise pygame.base.error, 'Invalid joystick axis'

        return SDL_JoystickGetAxis(self._device, axis) / 32768.0

    def get_numballs(self):
        '''Get the number of trackballs on a Joystick.

        Returns the number of trackball devices on a Joystick. These devices
        work similar to a mouse but they have no absolute position; they only
        have relative amounts of movement.

        The pygame.JOYBALLMOTION event will be sent when the trackball is
        rolled.  It will report the amount of movement on the trackball.

        :rtype: int
        '''
        self._init_check()
        return SDL_JoystickNumBalls(self._device)

    def get_ball(self, ball):
        '''Get the relative position of a trackball.

        Returns the relative movement of a joystick button. The value is a x,
        y pair holding the relative movement since the last call to get_ball.

        The ball number must be an integer from zero to get_numballs()-1.

        :Parameters:
            `ball` : int
                Ball to read.

        :rtype: int, int
        :return: relative X, relative Y.
        '''
        self._init_check()
        if ball < 0 or ball >= SDL_JoystickNumBalls(self._device):
            raise pygame.base.error, 'Invalid joystick trackball'

        return SDL_JoystickGetBall(self._device, ball)

    def get_numbuttons(self):
        '''Get the number of buttons on a Joystick.

        Returns the number of pushable buttons on the joystick. These buttons
        have a boolean (on or off) state.

        Buttons generate a pygame.JOYBUTTONDOWN and pygame.JOYBUTTONUP event
        when they are pressed and released.

        :rtype: int
        '''
        self._init_check()
        return SDL_JoystickNumButtons()

    def get_button(self, button):
        '''Get the current button state.

        Returns the current state of a joystick button.

        :Parameters:
            `button` : int
                Button to read.

        :rtype: bool
        '''
        self._init_check()
        if button < 0 or button >= SDL_JoystickNumButtons(self._device):
            raise pygame.base.error, 'Invalid joystick button'

        return SDL_JoystickGetButton(self._device, button)

    def get_numhats(self):
        '''Get the number of hat controls on a Joystick.

        Returns the number of joystick hats on a Joystick. Hat devices are
        like miniature digital joysticks on a joystick. Each hat has two axes
        of input.

        The pygame.JOYHATMOTION event is generated when the hat changes
        position.  The position attribute for the event contains a pair of
        values that are either -1, 0, or 1. A position of (0, 0) means the hat
        is centered.
        
        :rtype: int
        '''
        self._init_check()
        return SDL_JoystickNumHats()

    def get_hat(self, hat):
        '''Get the position of a joystick hat.

        Returns the current position of a position hat. The position is given
        as two values representing the X and Y position for the hat. (0, 0)
        means centered.  A value of -1 means left/down and a value of 1 means
        right/up: so (-1, 0) means left; (1, 0) means right; (0, 1) means up;
        (1, 1) means upper-right; etc.

        This value is digital, i.e., each coordinate can be -1, 0 or 1 but
        never in-between.

        The hat number must be between 0 and get_numhats()-1.

        :Parameters:
            `hat` : int
                Hat to read.

        :rtype: int, int
        :return: X, Y
        '''
        self._init_check()
        if hat < 0 or hat >= SDL_JoystickNumHats(self._device):
            raise pygame.base.error, 'Invalid joystick hat'

        value = SDL_JoystickGetHat(self._device, hat)
        x = y = 0
        if value & SDL_HAT_UP:
            y = 1
        elif value & SDL_HAT_DOWN:
            y = -1
        if value & SDL_HAT_RIGHT:
            x = 1
        elif value & SDL_HAT_LEFT:
            x = -1
        return x, y
