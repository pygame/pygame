#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import sys

from SDL import *
import pygame.base
import pygame.surface

_display_surface = None
_icon_was_set = 0

def __PYGAMEinit__():
    pygame.base.register_quit(_display_autoquit)

def _display_autoquit():
    global _display_surface
    _display_surface = None

def quit():
    pygame.base._video_autoquit()
    _display_autoquit()

def init():
    '''Initialize the display module.

    Manually initialize SDL's video subsystem.  Will raise an
    exception if it cannot be initialized. It is safe to call this
    function if the video has is currently initialized.
    '''
    pygame.base._video_autoinit()
    __PYGAMEinit__()

def get_init():
    '''Get status of display module initialization.

    :rtype: bool
    :return: True if SDL's video system is currently initialized.
    '''
    return SDL_WasInit(SDL_INIT_VIDEO) != 0

def get_active():
    '''get state of display mode

    Returns True if the current display is active on the screen. This
    done with the call to ``pygame.display.set_mode()``. It is
    potentially subject to the activity of a running window manager.

    Calling `set_mode` will change all existing display surface
    to reference the new display mode. The old display surface will
    be lost after this call.
    '''
    return SDL_GetAppState() & SDL_APPACTIVE != 0

class Info:
    def __init__(self):
        info = SDL_GetVideoInfo()
        if not info:
            raise pygame.base.error, 'Could not retrieve video info'
        self.hw = info.hw_available
        self.wm = info.wm_available
        self.blit_hw = info.blit_hw
        self.blit_hw_CC = info.blit_hw_CC
        self.blit_hw_A = info.blit_hw_A
        self.blit_sw = info.blit_sw
        self.blit_sw_CC = info.blit_sw_CC
        self.blit_sw_A = info.blit_sw_A
        self.blit_fill = info.blit_fill
        self.video_mem = info.video_mem
        self.bitsize = info.vfmt.BitsPerPixel
        self.bytesize = info.vfmt.BytesPerPixel
        self.masks = (info.vfmt.Rmask, info.vfmt.Gmask, 
                      info.vfmt.Bmask, info.vfmt.Amask)
        self.shifts = (info.vfmt.Rshift, info.vfmt.Gshift, 
                       info.vfmt.Bshift, info.vfmt.Ashift)
        self.losses = (info.vfmt.Rloss, info.vfmt.Gloss, 
                       info.vfmt.Bloss, info.vfmt.Aloss)

    def __str__(self):
        return ('<VideoInfo(hw = %d, wm = %d,video_mem = %d\n' + \
                '           blit_hw = %d, blit_hw_CC = %d, blit_hw_A = %d,\n'
                '           blit_sw = %d, blit_sw_CC = %d, blit_sw_A = %d,\n'
                '           bitsize  = %d, bytesize = %d,\n'
                '           masks =  (%d, %d, %d, %d),\n'
                '           shifts = (%d, %d, %d, %d),\n'
                '           losses =  (%d, %d, %d, %d)>\n') % \
               (self.hw, self.wm, self.video_mem,
                self.blit_hw, self.blit_hw_CC, self.blit_hw_A,
                self.blit_sw, self.blit_sw_CC, self.blit_sw_A,
                self.bitsize, self.bytesize,
                self.masks[0], self.masks[1], self.masks[2], self.masks[3],
                self.shifts[0], self.shifts[1], self.shifts[2], self.shifts[3],
                self.losses[0], self.losses[1], self.losses[2], self.losses[3])

    def __repr__(self):
        return str(self)

def get_driver():
    '''Get the current SDL video driver.

    Once the display is initialized, this will return the name of the
    currently running video driver. There is no way to get a list of
    all the supported video drivers.

    :rtype: str
    '''
    return SDL_VideoDriverName()

def get_wm_info():
    '''Get settings from the system window manager.

    :note: Currently unimplemented, returns an empty dict.
    :rtype: dict
    '''
    return {}

def get_surface():
    '''Get current display surface.

    Returns a `Surface` object representing the current display.  Will
    return None if called before the display mode is set.
    '''
    return _display_surface

def gl_set_attribute(flag, value):
    '''Set special OpenGL attributes.

    When calling `pygame.display.set_mode` with the OPENGL flag,
    pygame automatically handles setting the OpenGL attributes like
    color and doublebuffering. OpenGL offers several other attributes
    you may want control over. Pass one of these attributes as the
    flag, and its appropriate value.

    This must be called before `pygame.display.set_mode`.

    The OPENGL flags are: GL_ALPHA_SIZE, GL_DEPTH_SIZE, GL_STENCIL_SIZE,
    GL_ACCUM_RED_SIZE, GL_ACCUM_GREEN_SIZE, GL_ACCUM_BLUE_SIZE,
    GL_ACCUM_ALPHA_SIZE GL_MULTISAMPLEBUFFERS, GL_MULTISAMPLESAMPLES,
    GL_STEREO.

    :Parameters:
     - `flag`: int
     - `value`: int

    '''
    SDL_GL_SetAttribute(flag, value)

def gl_get_attribute(flag):
    '''Get special OpenGL attributes.

    After calling `pygame.display.set_mode` with the OPENGL flag
    you will likely want to check the value of any special OpenGL
    attributes you requested. You will not always get what you
    requested.

    See `gl_set_attribute` for a list of flags.

    :Parameters:
     - `flag`: int

    :rtype: int
    '''
    return SDL_GL_GetAttribute(flag)

def set_mode(size, flags=0, depth=0):
    '''Set the display mode.

    Sets the current display mode. If calling this after the mode has
    already been set, this will change the display mode to the
    desired type. Sometimes an exact match for the requested video
    mode is not available. In this case SDL will try to find the
    closest match and work with that instead.

    The size is a 2-number-sequence containing the width and height
    of the desired display mode. Flags represents a set of different
    options for the new display mode. If omitted or given as 0, it
    will default to a simple software window. You can mix several
    flags together with the bitwise-or (|) operator. Possible flags
    are HWSURFACE (or the value 1), HWPALETTE, DOUBLEBUF, and/or
    FULLSCREEN. There are other flags available but these are the
    most usual. A full list of flags can be found in the pygame
    documentation.

    The optional depth argument is the requested bits
    per pixel. It will usually be left omitted, in which case the
    display will use the best/fastest pixel depth available.

    You can create an OpenGL surface (for use with PyOpenGL)
    by passing the OPENGL flag. You will likely want to use the
    DOUBLEBUF flag when using OPENGL. In which case, the flip()
    function will perform the GL buffer swaps. When you are using
    an OPENGL video mode, you will not be able to perform most of the
    pygame drawing functions (fill, set_at, etc) on the display surface.

    :Parameters:
     - `size`: int, int
     - `flags`: int
     - `depth`: int
    '''
    global _display_surface

    w, h = size
    if w <= 0 or h <= 0:
        raise pygame.base.error, 'Cannot set 0 sized display mode'

    if not SDL_WasInit(SDL_INIT_VIDEO):
        init()

    if flags & SDL_OPENGL:
        if flags & SDL_DOUBLEBUF:
            flags &= ~SDL_DOUBLEBUF
            SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1)
        else:
            SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 0)
        if depth:
            SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, depth)
        surf = SDL_SetVideoMode(w, h, depth, flags)
        if SDL_GL_GetAttribute(SDL_GL_DOUBLEBUFFER):
            surf.flags |= SDL_DOUBLEBUF
    else:
        if not depth:
            flags |= SDL_ANYFORMAT
        surf = SDL_SetVideoMode(w, h, depth, flags)
        title, icontitle = SDL_WM_GetCaption()
        if not title:
            SDL_WM_SetCaption('pygame window', 'pygame')

        SDL_PumpEvents()

        if _display_surface:
            _display_surface._surf = surf
        else:
            _display_surface = pygame.surface.Surface(surf=surf)

    if sys.platform != 'darwin' and False: # XXX
        if not _icon_was_set:
            iconsurf = _display_resource(_icon_defaultname)
            SDL_SetColorKey(iconsurf.surf, SDL_SRCCOLORKEY, 0)
            _do_set_icon(iconsurf)

    return _display_surface
