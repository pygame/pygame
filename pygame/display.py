#!/usr/bin/env python

'''Pygame module to control the display window and screen.

This module offers control over the pygame display. Pygame has a single display
Surface that is either contained in a window or runs full screen. Once you
create the display you treat it as a regular Surface. Changes are not
immediately visible onscreen, you must choose one of the two flipping functions
to update the actual display.

The pygame display can actually be initialized in one of several modes. By
default the display is a basic software driven framebuffer. You can request
special modules like hardware acceleration and OpenGL support. These are
controlled by flags passed to pygame.display.set_mode().

Pygame can only have a single display active at any time. Creating a new one
with pygame.display.set_mode() will close the previous display. If precise
control is needed over the pixel format or display resolutions, use the
functions pygame.display.mode_ok(), pygame.display.list_modes(), and
pygame.display.Info() to query information about the display.

Once the display Surface is created, the functions from this module
effect the single existing display. The Surface becomes invalid if the module
is uninitialized. If a new display mode is set, the existing Surface will
automatically switch to operate on the new display.

Then the display mode is set, several events are placed on the pygame
event queue. pygame.QUIT is sent when the user has requested the program
to shutdown. The window will receive pygame.ACTIVEEVENT events as the
display gains and loses input focus. If the display is set with the
pygame.RESIZABLE flag, pygame.VIDEORESIZE events will be sent when the
user adjusts the window dimensions. Hardware displays that draw direct
to the screen will get pygame.VIDEOEXPOSE events when portions of the
window must be redrawn.
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

def init():
    '''Initialize the display module.

    Initializes the pygame display module. The display module cannot do
    anything until it is initialized. This is usually handled for you
    automatically when you call the higher level `pygame.init`.

    Pygame will select from one of several internal display backends when it
    is initialized. The display mode will be chosen depending on the platform
    and permissions of current user. Before the display module is initialized
    the environment variable SDL_VIDEODRIVER can be set to control which
    backend is used. The systems with multiple choices are listed here.

    Windows
        windib, directx
    Unix
        x11, dga, fbcon, directfb, ggi, vgl, svgalib, aalib

    On some platforms it is possible to embed the pygame display into an already
    existing window. To do this, the environment variable SDL_WINDOWID must be
    set to a string containing the window id or handle. The environment variable
    is checked when the pygame display is initialized. Be aware that there can
    be many strange side effects when running in an embedded display.

    It is harmless to call this more than once, repeated calls have no effect.
    '''
    pygame.base._video_autoinit()
    __PYGAMEinit__()

def quit():
    '''Uninitialize the display module.

    This will shut down the entire display module. This means any active
    displays will be closed. This will also be handled automatically when the
    program exits.

    It is harmless to call this more than once, repeated calls have no effect.
    '''
    pygame.base._video_autoquit()
    _display_autoquit()

def get_init():
    '''Get status of display module initialization.

    :rtype: bool
    :return: True if SDL's video system is currently initialized.
    '''
    return SDL_WasInit(SDL_INIT_VIDEO) != 0

def set_mode(resolution, flags=0, depth=0):
    '''Initialize a window or screen for display.

    This function will create a display Surface. The arguments passed in are
    requests for a display type. The actual created display will be the best
    possible match supported by the system.

    The `resolution` argument is a pair of numbers representing the width and
    height. The `flags` argument is a collection of additional options.
    The `depth` argument represents the number of bits to use for color.

    The Surface that gets returned can be drawn to like a regular Surface but
    changes will eventually be seen on the monitor.

    It is usually best to not pass the depth argument. It will default to the
    best and fastest color depth for the system. If your game requires a
    specific color format you can control the depth with this argument. Pygame
    will emulate an unavailable color depth which can be slow.

    When requesting fullscreen display modes, sometimes an exact match for the
    requested resolution cannot be made. In these situations pygame will select
    the closest compatable match. The returned surface will still always match
    the requested resolution.

    The flags argument controls which type of display you want. There are
    several to choose from, and you can even combine multiple types using the
    bitwise or operator, (the pipe "|" character). If you pass 0 or no flags
    argument it will default to a software driven window. Here are the display
    flags you will want to choose from:

    pygame.FULLSCREEN
        create a fullscreen display
    pygame.DOUBLEBUF
        recommended for HWSURFACE or OPENGL
    pygame.HWSURFACE
        hardware accelereated, only in FULLSCREEN
    pygame.OPENGL
        create an opengl renderable display
    pygame.RESIZABLE
        display window should be sizeable
    pygame.NOFRAME
        display window will have no border or controls

    :Parameters:
     - `resolution`: int, int
     - `flags`: int
     - `depth`: int

    :rtype: `Surface`
    '''
    global _display_surface

    w, h = resolution
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

def get_surface():
    '''Get current display surface.

    Returns a `Surface` object representing the current display.  Will
    return None if called before the display mode is set.

    :rtype: `Surface`
    '''
    return _display_surface

def flip():
    '''Update the full display surface to the screen.

    This will update the contents of the entire display. If your display mode
    is using the flags pygame.HWSURFACE and pygame.DOUBLEBUF, this will wait
    for a vertical retrace and swap the surfaces. If you are using a different
    type of display mode, it will simply update the entire contents of the
    surface.
 
    When using an pygame.OPENGL display mode this will perform a gl buffer
    swap.
    '''
    screen = SDL_GetVideoSurface()
    if not screen:
        raise pygame.base.error, 'Display mode not set'

    if screen.flags & SDL_OPENGL:
        SDL_GL_SwapBuffers()
    else:
        SDL_Flip(screen)

def _crop_rect(w, h, rect):
    if rect.x >= w or rect.y >= h or \
       rect.x + rect.w <= 0 or rect.y + rect.h <= 0:
        return None
    rect.x = max(rect.x, 0)
    rect.y = max(rect.y, 0)
    rect.w = min(rect.x + rect.w, w) - rect.x 
    rect.h = min(rect.y + rect.h, h) - rect.y 
    return rect

def update(*rectangle):
    '''Update portions of the screen for software displays.

    This function is like an optimized version of pygame.display.flip() for
    software displays. It allows only a portion of the screen to updated,
    instead of the entire area. If no argument is passed it updates the entire
    Surface area like `flip`.

    You can pass the function a single rectangle, or a sequence of rectangles.
    It is more efficient to pass many rectangles at once than to call update
    multiple times with single or a partial list of rectangles. If passing
    a sequence of rectangles it is safe to include None values in the list,
    which will be skipped.

    This call cannot be used on pygame.OPENGL displays and will generate an
    exception.

    :Parameters:
        `rectangle` : Rect or sequence of Rect
            Area(s) to update

    '''
    # Undocumented: also allows argument tuple to represent one rect;
    # e.g. update(0, 0, 10, 10) or update((0, 0), (10, 10))

    screen = SDL_GetVideoSurface()
    if not screen:
        raise pygame.base.error, 'Display mode not set'

    if screen.flags & SDL_OPENGL:
        raise pygame.base.error, 'Cannot update an OPENGL display'

    if not rectangle:
        SDL_UpdateRect(screen, 0, 0, 0, 0)
    else:
        w, h = screen.w, screen.h
        try:
            rect = pygame.rect._rect_from_object(rectangle)._r
            rect = _crop_rect(w, h, rect)
            if rect:
                SDL_UpdateRect(screen, rect.x, rect.y, rect.w, rect.h)
        except TypeError:
            rectangle = rectangle[0]
            rects = [_crop_rect(w, h, pygame.rect._rect_from_object(r)._r) \
                     for r in rectangle if r]
            SDL_UpdateRects(screen, rects) 
            

def get_driver():
    '''Get name of the pygame display backend.

    Pygame chooses one of many available display backends when it is
    initialized.  This returns the internal name used for the display backend.
    This can be used to provide limited information about what display
    capabilities might be accelerated.

    :rtype: str
    '''
    return SDL_VideoDriverName()

def Info():
    '''Create a video display information object.

    Creates a simple object containing several attributes to describe the
    current graphics environment. If this is called before
    `set_mode` some platforms can provide information about the default
    display mode. This can also be called after setting the display mode to
    verify specific display options were satisfied. 

    :see: `VideoInfo`
    :rtype: `VideoInfo`
    '''
    return VideoInfo()

class VideoInfo:
    '''Video display information.

    :Ivariables:
        `hw` : bool
            True if the display is hardware accelerated.
        `wm` : bool
            True if windowed display modes can be used.
        `video_mem` : int
            The amount of video memory on the displaoy, in megabytes.  0 if
            unknown.
        `bitsize` : int
            Number of bits used to store each pixel.
        `bytesize` : int
            Number of bytes used to store each pixel.
        `masks` : (int, int, int, int)
            RGBA component mask.
        `shifts` : (int, int, int, int)
            RGBA component shift amounts.
        `losses` : (int, int, int, int)
            Number of bits lost from a 32 bit depth for each RGBA component.
        `blit_hw` : bool
            True if hardware Surface blitting is accelerated
        `blit_hw_CC` : bool
            True if hardware Surface colorkey blitting is accelerated
        `blit_hw_A` : bool
            True if hardware Surface pixel alpha blitting is accelerated
        `blit_sw` : bool
            True if software Surface blitting is accelerated
        `blit_sw_CC` : bool
            True if software Surface colorkey blitting is accelerated
        `blit_sw_A` : bool
            True if software Surface pixel alpha blitting is acclerated

    '''

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

def get_wm_info():
    '''Get settings from the system window manager.

    :note: Currently unimplemented, returns an empty dict.
    :rtype: dict
    '''
    return {}

def list_modes(depth=0, flags=pygame.constants.FULLSCREEN):
    '''Get list of available fullscreen modes.

    This function returns a list of possible dimensions for a specified color
    depth. The return value will be an empty list if no display modes are
    available with the given arguments. A return value of -1 means that any
    requested resolution should work (this is likely the case for windowed
    modes). Mode sizes are sorted from biggest to smallest.
     
    If depth is 0, SDL will choose the current/best color depth for the
    display.  The flags defaults to pygame.FULLSCREEN, but you may need to add
    additional flags for specific fullscreen modes.

    :rtype: list of (int, int), or -1
    :return: list of (width, height) pairs, or -1 if any mode is suitable.
    '''
    format = SDL_PixelFormat()
    format.BitsPerPixel = depth

    if not format.BitsPerPixel:
        format.BitsPerPixel = SDL_GetVideoInfo().vfmt.BitsPerPixel

    rects = SDL_ListModes(format, flags)

    if rects == -1:
        return -1

    return [(r.w, r.h) for r in rects]

def mode_ok(size, flags=0, depth=0):
    '''Pick the best color depth for a display mode

    This function uses the same arguments as pygame.display.set_mode(). It is
    used to depermine if a requested display mode is available. It will return
    0 if the display mode cannot be set. Otherwise it will return a pixel
    depth that best matches the display asked for.

    Usually the depth argument is not passed, but some platforms can support
    multiple display depths. If passed it will hint to which depth is a better
    match.

    The most useful flags to pass will be pygame.HWSURFACE, pygame.DOUBLEBUF, 
    and maybe pygame.FULLSCREEN. The function will return 0 if these display
    flags cannot be set.

    :rtype: int
    :return: depth, in bits per pixel, or 0 if the requested mode cannot be
        set.
    '''
    if not depth:
        depth = SDL_GetVideoInfo().vfmt.BitsPerPixel
    return SDL_VideoModeOK(size[0], size[1], depth, flags)

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

def get_active():
    '''Get state of display mode

    Returns True if the current display is active on the screen. This
    done with the call to ``pygame.display.set_mode()``. It is
    potentially subject to the activity of a running window manager.

    Calling `set_mode` will change all existing display surface
    to reference the new display mode. The old display surface will
    be lost after this call.
    '''
    return SDL_GetAppState() & SDL_APPACTIVE != 0

def iconify():
    '''Iconify the display surface.

    Request the window for the display surface be iconified or hidden. Not all
    systems and displays support an iconified display. The function will
    return True if successfull.

    When the display is iconified pygame.display.get_active() will return
    False. The event queue should receive a pygame.APPACTIVE event when the
    window has been iconified.

    :rtype: bool
    :return: True on success
    '''
    try:
        SDL_WM_IconifyWindow()
        return True
    except SDL_Exception:
        return False

def toggle_fullscreen():
    '''Switch between fullscreen and windowed displays.

    Switches the display window between windowed and fullscreen modes. This
    function only works under the unix x11 video driver. For most situations
    it is better to call pygame.display.set_mode() with new display flags.

    :rtype: bool
    '''
    screen = SDL_GetVideoSurface()
    try:
        SDL_WM_ToggleFullScreen(screen)
        return True
    except SDL_Exception:
        return False

def set_gamma(red, green=None, blue=None):
    '''Change the hardware gamma ramps.

    Set the red, green, and blue gamma values on the display hardware. If the
    green and blue arguments are not passed, they will both be the same as
    red.  Not all systems and hardware support gamma ramps, if the function
    succeeds it will return True.

    A gamma value of 1.0 creates a linear color table. Lower values will
    darken the display and higher values will brighten.

    :Parameters:
        `red` : float
            Red gamma value
        `green` : float
            Green gamma value
        `blue` : float
            Blue gamma value

    :rtype: bool
    '''
    if not green or not blue:
        green = red
        blue = red

    try:
        SDL_SetGamma(red, green, blue)
        return True
    except SDL_Exception:
        return False

def set_gamma_ramp(red, green, blue):
    '''Change the hardware gamma ramps with a custom lookup.

    Set the red, green, and blue gamma ramps with an explicit lookup table.
    Each argument should be sequence of 256 integers. The integers should
    range between 0 and 0xffff. Not all systems and hardware support gamma
    ramps, if the function succeeds it will return True.

    :Parameters:
        `red` : sequence of int
            Sequence of 256 ints in range [0, 0xffff] giving red component
            lookup.
        `green` : sequence of int
            Sequence of 256 ints in range [0, 0xffff] giving green component
            lookup.
        `blue` : sequence of int
            Sequence of 256 ints in range [0, 0xffff] giving blue component
            lookup.

    :rtype: bool
    '''
    try:
        SDL_SetGammaRamp(red, green, blue)
        return True
    except SDL_Exception:
        return False

def set_icon(surface):
    '''Change the system image for the display window.

    Sets the runtime icon the system will use to represent the display window.
    All windows default to a simple pygame logo for the window icon.

    You can pass any surface, but most systems want a smaller image around
    32x32.  The image can have colorkey transparency which will be passed to
    the system.

    Some systems do not allow the window icon to change after it has been
    shown. This function can be called before `set_mode` to
    create the icon before the display mode is set.

    :Parameters:
        `surface` : `Surface`
            Surface containing image to set.

    '''
    global _icon_was_set

    pygame.base._video_autoinit()
    SDL_WM_SetIcon(surface._surf, None)
    _icon_was_set = 1

def set_caption(title, icontitle=None):
    '''Set the current window caption.

    If the display has a window title, this function will change the name on
    the window. Some systems support an alternate shorter title to be used for
    minimized displays.

    :Parameters:
        `title` : unicode
            Window caption
        `icontitle` : unicode
            Icon caption, if supported

    '''
    if not icontitle:
        icontitle = title

    SDL_WM_SetCaption(title, icontitle)

def get_caption():
    '''Get the current window caption.

    Returns the title and icontitle for the display Surface. These will often
    be the same value.

    :rtype: (unicode, unicode)
    :return: title, icontitle
    '''
    # XXX deviation from pygame, don't return () if title == None
    return SDL_WM_GetCaption()

def set_palette(palette=None):
    '''Set the display color palette for indexed displays.

    This will change the video display color palette for 8bit displays. This
    does not change the palette for the actual display Surface, only the
    palette that is used to display the Surface. If no palette argument is
    passed, the system default palette will be restored. The palette is a
    sequence of RGB triplets.

    :Parameters:
        `palette` : sequence of (int, int, int)
            Sequence having at most 256 RGB triplets.

    '''
    surf = SDL_GetVideoSurface()
    if not surf:
        raise pygame.base.error, 'No display mode is set'
    if surf.format.BytesPerPixel != 1 or not surf.format._palette:
        raise pygame.base.error, 'Display mode is not colormapped'

    if not palette:
        SDL_SetPalette(surf, SDL_PHYSPAL, surf.format.palette.colors, 0)

    lenth = min(surf.format.palette.ncolors, len(palette))
    colors = [SDL_Color(r, g, b) for r, g, b in palette[:length]]
    SDL_SetPalette(surf, SDL_PHYSPAL, colors, 0)
