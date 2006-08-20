#!/usr/bin/env python

'''Pygame module to work with the mouse.

The mouse functions can be used to get the current state of the mouse device.
These functions can also alter the system cursor for the mouse.

When the display mode is set, the event queue will start receiving mouse
events. The mouse buttons generate pygame.MOUSEBUTTONDOWN and
pygame.MOUSEBUTTONUP events when they are pressed and released. These
events contain a button attribute representing which button was pressed.
The mouse wheel will generate pygame.MOUSEBUTTONDOWN events when rolled.
The button will be set to 4 when the wheel is rolled up, and to button
5 when the wheel is rolled down. Anytime the mouse is moved it generates
a pygame.MOUSEMOTION event. The mouse movement is broken into small
and accurate motion events. As the mouse is moving many motion events will
be placed on the queue. Mouse motion events that are not properly cleaned
from the event queue are the primary reason the event queue fills up.

If the mouse cursor is hidden, and input is grabbed to the current display
the mouse will enter a virtual input mode, where the relative movements
of the mouse will never be stopped by the borders of the screen. See the
functions pygame.mouse.set_visible() and pygame.event.set_grab() to get
this configured.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from SDL import *
import pygame.base
import pygame.display

def get_pressed():
    '''Get the state of the mouse buttons.

    Returns a sequence of booleans representing the state of all the mouse
    buttons. A true value means the mouse is currently being pressed at the
    time of the call.

    Note, to get all of the mouse events it is better to use either 
    pygame.event.wait() or pygame.event.get() and check all of those events
    to see if they are MOUSEBUTTONDOWN, MOUSEBUTTONUP, or MOUSEMOTION.

    Note, that on X11 some XServers use middle button emulation.  When you click
    both buttons 1 and 3 at the same time a 2 button event can be emitted.

    Note, remember to call pygame.event.get() before this function.  Otherwise 
    it will not work.
    
    :rtype: int, int, int
    :return: left, middle, right
    '''
    pygame.display._video_init_check()

    state, x, y = SDL_GetMouseState()
    return state & SDL_BUTTON(1), state & SDL_BUTTON(2), state & SDL_BUTTON(3)

def get_pos():
    '''Get the mouse cursor position.

    Returns the X and Y position of the mouse cursor. The position is relative
    the the top-left corner of the display. The cursor position can be located
    outside of the display window, but is always constrained to the screen.
    
    :rtype: int, int
    :return: X, Y
    '''
    pygame.display._video_init_check()

    state, x, y = SDL_GetMouseState()
    return x, y

def get_rel():
    '''Get the most recent amount of mouse movement.

    Returns the amount of movement in X and Y since the previous call to this
    function. The relative movement of the mouse cursor is constrained to the
    edges of the screen, but see the virtual input mouse mode for a way around
    this.  Virtual input mode is described at the top of the page.
    
    :rtype: int, int
    :return: X, Y
    '''
    pygame.display._video_init_check()

    return SDL_GetRelativeMouseState()

def set_pos(x, y):
    '''Set the mouse cursor position.

    Set the current mouse position to arguments given. If the mouse cursor is
    visible it will jump to the new coordinates. Moving the mouse will generate
    a new pygame.MOUSEMOTION event.
    
    :Parameters:
        `x` : int
            X coordinate of mouse cursor
        `y` : int
            Y coordinate of mouse cursor

    '''
    pygame.display._video_init_check()

    SDL_WarpMouse(x, y)

def set_visible(visible):
    '''Hide or show the mouse cursor.

    If the `visible` argument is true, the mouse cursor will be visible. This
    will return the previous visible state of the cursor.
    
    :Parameters:
        `visible` : bool
            If True, the mouse cursor will be visible.

    '''
    pygame.display._video_init_check()

    # XXX Differ from pygame: no return value here
    SDL_ShowCursor(visible)

def get_focused():
    '''Check if the display is receiving mouse input.

    Returns true when pygame is receiving mouse input events
    (or, in windowing terminology, is "active" or has the "focus").

    This method is most useful when working in a window.
    By contrast, in full-screen mode, this method
    always returns true.

    Note: under MS Windows, the window that has the mouse focus
    also has the keyboard focus. But under X-Windows, one window
    can receive mouse events and another receive keyboard events.
    pygame.mouse.get_focused() indicates whether the pygame
    window receives mouse events.
    
    :rtype: bool
    '''
    pygame.display._video_init_check()

    return SDL_GetAppState() & SDL_APPMOUSEFOCUS != 0

def set_cursor(size, hotspot, xormask, andmask):
    '''Set the image for the system mouse cursor.

    When the mouse cursor is visible, it will be displayed as a black and white
    bitmap using the given bitmask arrays. The size is a sequence containing
    the cursor width and height. Hotspot is a sequence containing the cursor
    hotspot position. xormasks is a sequence of bytes containing the cursor
    xor data masks. Lastly is andmasks, a sequence of bytes containting the
    cursor bitmask data.
     
    Width must be a multiple of 8, and the mask arrays must be the correct
    size for the given width and height. Otherwise an exception is raised.

    See the pygame.cursor module for help creating default and custom 
    masks for the system cursor.
    
    :Parameters:
        `size` : int, int
            Tuple of (width, height)
        `hotspot` : int, int
            Tuple of (X, Y)
        `xormask` : list of int
            Bitmask of color: white if 0, black if 1.
        `andmask` : list of int
            Bitmask of mask: transparent if 0, solid if 1

    '''
    pygame.display._video_init_check()

    w, h = size
    spotx, spoty = hotspot
    
    if w % 8 != 0:
        raise ValueError, 'Cursor width must be divisible by 8.'

    if len(xormask) != w * h / 8 or len(andmask) != w * h / 8:
        raise ValueError, 'bitmasks must be sized width*height/8'

    xormask = [int(i) for i in xormask]
    andmask = [int(i) for i in andmask]

    cursor = SDL_CreateCursor(xormask, andmask, w, h, spotx, spoty)

    lastcursor = SDL_GetCursor()
    SDL_SetCursor(cursor)
    SDL_FreeCursor(lastcursor)

def get_cursor():
    '''Get the image for the system mouse cursor.

    Get the information about the mouse system cursor. The return value is the
    same data as the arguments passed into pygame.mouse.set_cursor().

    :rtype: tuple, tuple, list, list
    :return: size, hotspot, xormasks, andmasks
    '''
    pygame.display._video_init_check()

    cursor = SDL_GetCursor()

    w = cursor.area.w
    h = cursor.area.h
    spotx = cursor.hot_x
    spoty = cursor.hot_y

    xordata = list(cursor.data)
    anddata = list(cursor.mask)

    return (w, h), (spotx, spoty), xordata, anddata
