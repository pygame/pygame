#!/usr/bin/env python

'''pygame module to work with the keyboard

This module contains functions for dealing with the keyboard.


The event queue gets pygame.KEYDOWN and pygame.KEYUP events when the
keyboard buttons are pressed and released. Both events have a key attribute
that is a integer id representing every key on the keyboard. 
The pygame.KEYDOWN event has an additional attribute named unicode. This
represents a single character string that is the fully translated character
entered. This takes into account the shift and composition keys.

The keyboard also has a list of modifier states that can be assembled
bit bitwise ORing them together.
 
      KMOD_NONE, KMOD_LSHIFT, KMOD_RSHIFT, KMOD_SHIFT, KMOD_CAPS,
      KMOD_LCTRL, KMOD_RCTRL, KMOD_CTRL, KMOD_LALT, KMOD_RALT,
      KMOD_ALT, KMOD_LMETA, KMOD_RMETA, KMOD_META, KMOD_NUM, KMOD_MODE
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from SDL import *
import pygame.display

def get_focused():
    '''Determine if the display is receiving keyboard input from the system.

    This is true when the display window has keyboard focus from the system.
    If the display needs to ensure it does not lose keyboard focus, it can
    use pygame.event.set_grab() to grab all input.

    :rtype: bool
    '''
    pygame.display._video_init_check()

    return SDL_GetAppState() & SDL_APPINPUTFOCUS != 0

def get_pressed():
    '''Get the state of all keyboard buttons.

    Returns a sequence of boolean values representing the state of every
    key on the keyboard. Use the key constant values to index the array. A
    True value means the that button is pressed.

    Getting the list of pushed buttons with this function is not the proper
    way to handle text entry from the user. You have no way to know the order
    of keys pressed, and rapidly pushed keys can be completely unnoticed
    between two calls to pygame.key.get_pressed(). There is also no way to
    translate these pushed keys into a fully translated character value.
    See the pygame.KEYDOWN events on the event queue for this functionality.

    :rtype: list of bool
    '''
    pygame.display._video_init_check()

    return SDL_GetKeyState()

def get_mods():
    '''Determine which modifier keys are being held.

    Returns a single integer representing a bitmask of all the modifier
    keys being held. Using bitwise operators you can test if specific shift
    keys are pressed, the state of the capslock button, and more.

    :rtype: int
    '''
    pygame.display._video_init_check()

    return SDL_GetModeState()

def set_mods(mods):
    '''Temporarily set which modifier keys are pressed.

    Create a bitmask of the modifier constants you want to impose on your
    program.

    :Parameters:
        `mods` : int
            Bitmask of modifier constants.

    '''
    pygame.display._video_init_check()

    SDL_SetModState(mods)

def set_repeat(delay=0, interval=0):
    '''Control how held keys are repeated.

    When the keyboard repeat is enabled, keys that are held down will generate
    multiple pygame.KEYDOWN events. The delay is the number if milliseconds
    before the first repeated pygame.KEYDOWN will be sent. After that another
    pygame.KEYDOWN will be sent every interval milliseconds. If no arguments
    are passed the key repeat is disabled.

    When pygame is initialized the key repeat is disabled.

    :Parameters:
        `delay` : int
            Time in milliseconds before key repeat is activated
        `interval` : int
            Time in milliseconds between repeated key events.

    '''
    pygame.display._video_init_check()
    
    if delay and not interval:
        interval = delay

    SDL_EnableKeyRepeat(delay, interval)

def name(key):
    '''Get the name of a key identifier.

    Get the descriptive name of the button from a keyboard button id constant.

    :Parameters:
        `key` : int
            Key symbol constant (e.g., K_F5).

    :rtype: str
    '''
    return SDL_GetKeyName(key)
