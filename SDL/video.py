#!/usr/bin/env python

'''Access to the SDL raw framebuffer window.

:group YUV overlay functions: SDL_CreateYUVOverlay, SDL_LockYUVOverlay,
    SDL_UnlockYUVOverlay, SDL_DisplayYUVOverlay, SDL_FreeYUVOverlay

:group OpenGL support functions: SDL_GL_SetAttribute,
    SDL_GL_GetAttribute, SDL_GL_SwapBuffers

:group Window manager functions: SDL_WM_SetCaption, SDL_WM_GetCaption,
    SDL_WM_SetIcon, SDL_WM_IconifyWindow, SDL_WM_ToggleFullScreen,
    SDL_WM_GrabInput
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.array
import SDL.dll
import SDL.error
import SDL.constants
import SDL.rwops

class SDL_Rect(Structure):
    '''Rectangle structure.

    Rectangles must be normalised, i.e., with their width and height
    greater than zero.

    :Ivariables:
        `x` : int
            Top-left x coordinate.
        `y` : int
            Top-left y coordinate.
        `w` : int
            Width
        `h` : int
            Height

    '''

    _fields_ = [('x', c_short),
                ('y', c_short),
                ('w', c_ushort),
                ('h', c_ushort)]

    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def __repr__(self):
        return 'SDL_Rect(x=%d, y=%d, w=%d, h=%d)' % \
            (self.x, self.y, self.w, self.h)

    def __copy__(self):
        return SDL_Rect(self.x, self.y, self.w, self.h)

    def __deepcopy__(self, memo):
        return SDL_Rect(self.x, self.y, self.w, self.h)

class SDL_Color(Structure):
    '''Structure defining a single color.

    ``SDL_Colour`` is a synonym for ``SDL_Color``.

    :Ivariables:
        `r` : int
            Red component, in range [0,255]
        `g` : int
            Green component, in range [0,255]
        `b` : int
            Blue component, in range [0,255]

    '''
    _fields_ = [('r', c_ubyte),
                ('g', c_ubyte),
                ('b', c_ubyte),
                ('unused', c_ubyte)]

    def __init__(self, r=0, g=0, b=0, unused=0):
        self.r = r
        self.g = g
        self.b = b

    def __repr__(self):
        return 'SDL_Color(r=%d, g=%d, b=%d)' % \
            (self.r, self.g, self.b)

    def __copy__(self):
        return SDL_Color(self.r, self.g, self.b)

    def __deepcopy__(self, memo):
        return SDL_Color(self.r, self.g, self.b)

SDL_Colour = SDL_Color

class SDL_Palette(Structure):
    '''Color palette array.

    :Ivariables:
        `ncolors` : int
            Number of colors in the palette
        `colors` : `SDL_array`
            Array of indexed colors of type `SDL_Color`.

    '''
    _fields_ = [('ncolors', c_int),
                ('_colors', POINTER(SDL_Color))]

    def __getattr__(self, name):
        if name == 'colors':
            return SDL.array.SDL_array(self._colors, self.ncolors, SDL_Color)
        raise AttributeError, name
        

class SDL_PixelFormat(Structure):
    '''Read-only surface format structure.

    :Ivariables:
        `BitsPerPixel` : int
            Number of bits per pixel
        `BytesPerPixel` : int
            Number of bytes per pixel.  This is not necessarily 8 *
            BitsPerPixel.
        `Rloss` : int
            Number of bits lost from an 8-bit red component
        `Gloss` : int
            Number of bits lost from an 8-bit green component
        `Bloss` : int
            Number of bits lost from an 8-bit blue component
        `Aloss` : int
            Number of bits lost from an 8-bit alpha component
        `Rshift` : int
            Number of bits to shift right red component when packing
        `Gshift` : int
            Number of bits to shift right green component when packing
        `Bshift` : int
            Number of bits to shift right blue component when packing
        `Ashift` : int
            Number of bits to shift right alpha component when packing
        `Rmask` : int
            32-bit mask of red component
        `Gmask` : int
            32-bit mask of green component
        `Bmask` : int
            32-bit mask of blue component
        `Amask` : int
            32-bit mask of alpha component
        `colorkey` : int
            Packed transparent color key, if set.
        `alpha` : int
            Surface alpha, in range [0, 255]

    '''
    _fields_ = [('_palette', POINTER(SDL_Palette)),
                ('BitsPerPixel', c_ubyte),
                ('BytesPerPixel', c_ubyte),
                ('Rloss', c_ubyte),
                ('Gloss', c_ubyte),
                ('Bloss', c_ubyte),
                ('Aloss', c_ubyte),
                ('Rshift', c_ubyte),
                ('Gshift', c_ubyte),
                ('Bshift', c_ubyte),
                ('Ashift', c_ubyte),
                ('Rmask', c_uint),
                ('Gmask', c_uint),
                ('Bmask', c_uint),
                ('Amask', c_uint),
                ('colorkey', c_uint),
                ('alpha', c_ubyte)]

    def __getattr__(self, name):
        if name == 'palette':
            if self._palette:
                return self._palette.contents
            return None
        raise AttributeError

    def __copy__(self):
        f = SDL_PixelFormat()
        f._palette = self._palette
        f.BitsPerPixel = self.BitsPerPixel
        f.BytesPerPixel = self.BytesPerPixel
        f.Rloss = self.Rloss
        f.Gloss = self.Gloss
        f.Bloss = self.Bloss
        f.Aloss = self.Aloss
        f.Rshift = self.Rshift
        f.Gshift = self.Gshift
        f.Bshift = self.Bshift
        f.Ashift = self.Ashift
        f.Rmask = self.Rmask
        f.Gmask = self.Gmask
        f.Bmask = self.Bmask
        f.Amask = self.Amask
        f.colorkey = self.colorkey
        f.alpha = self.alpha
        return f

class SDL_Surface(Structure):
    '''Read-only surface structure.

    :Ivariables:
        `format` : `SDL_PixelFormat`
            Pixel format used by this surface
        `w` : int
            Width
        `h` : int
            Height
        `pitch` : int
            Number of bytes between consecutive rows of pixel data.  Note
            that this may be larger than the width.
        `pixels` : SDL_array
            Buffer of integers of the type given in the pixel format. 
            Each element of the array corresponds to exactly one pixel
            when ``format.BitsPerPixel`` is 8, 16 or 32; otherwise each
            element is exactly one byte.  Modifying this array has an
            immediate effect on the pixel data of the surface.  See
            `SDL_LockSurface`.

    '''
    _fields_ = [('flags', c_uint),
                ('_format', POINTER(SDL_PixelFormat)),
                ('w', c_int),
                ('h', c_int),
                ('pitch', c_short),
                ('_pixels', POINTER(c_ubyte)),
                ('_offset', c_int),
                ('_hwdata', c_void_p),
                ('clip_rect', SDL_Rect),
                ('_unused1', c_int),
                ('_locked', c_int),
                ('_map', c_void_p),  
                ('_format_version', c_int),
                ('_refcount', c_int)]

    def __getattr__(self, name):
        if name == 'format':
            # Dereference format ptr
            return self._format.contents
        elif name == 'pixels':
            # Return SDL_array type for pixels
            if not self._pixels:
                raise SDL.error.SDL_Exception, 'Surface needs locking'
            bpp = self.format.BitsPerPixel
            count = self.pitch / self.format.BytesPerPixel * self.h
            if bpp == 1:
                sz = c_ubyte
                # Can't rely on BytesPerPixel when calculating 1 bit pitch
                byte_pitch = self.pitch  * 8 / bpp
                count = (byte_pitch * self.h + 7) / 8
            elif bpp == 8:
                sz = c_ubyte
            elif bpp == 16:
                sz = c_ushort
            elif bpp == 24:
                sz = c_ubyte
                count = self.pitch * self.h
            elif bpp == 32:
                sz = c_uint
            else:
                raise SDL.error.SDL_Exception, 'Unsupported bytes-per-pixel'
            return SDL.array.SDL_array(self._pixels, count, sz)
        raise AttributeError, name

def SDL_MUSTLOCK(surface):
    '''Evaluates to true if the surface needs to be locked before access.

    :Parameters:
      - `surface`: SDL_Surface

    :return: True if the surface needs to be locked before access,
             otherwise False.
    '''
    return surface._offset or \
        ((surface.flags & \
          (SDL.constants.SDL_HWSURFACE | \
           SDL.constants.SDL_ASYNCBLIT | \
           SDL.constants.SDL_RLEACCEL)) != 0)

SDL_blit = CFUNCTYPE(c_int, POINTER(SDL_Surface), POINTER(SDL_Rect),
                            POINTER(SDL_Surface), POINTER(SDL_Rect))

class SDL_VideoInfo(Structure):
    '''Useful for determining the video hardware capabilities.

    :Ivariables:
        `hw_available` : bool
            True if hardware surfaces can be created
        `wm_available` : bool
            True if a window manager is available
        `blit_hw` : bool
            True if hardware to hardware blits are accelerated
        `blit_hw_CC` : bool
            True if hardware to hardware color-keyed blits are accelerated
        `blit_hw_A` : bool
            True if hardware to hardware alpha-blended blits are accelerated
        `blit_sw` : bool
            True if software to hardware blits are accelerated
        `blit_sw_CC` : bool
            True if software to hardware color-keyed blits are accelerated
        `blit_sw_A` : bool
            True if software to hardware alpha-blended blits are accelerated
        `blit_fill` : bool
            True if color fills are accelerated
        `video_mem` : int
            Total amount of video memory, in kilobytes (unreliable)
        `vfmt` : `SDL_PixelFormat`
            Pixel format of the video surface
        `current_w` : int
            Current video width.  Available in SDL 1.2.10 and later.
        `current_h` : int
            Current video height.  Available in SDL 1.2.10 and later.

    '''
    _fields_ = [('bitfield', c_uint),
                ('video_mem', c_uint),
                ('_vfmt', POINTER(SDL_PixelFormat)),
                ('_current_w', c_int),
                ('_current_h', c_int)]

    def __getattr__(self, name):
        '''Retrieve bitfields as bool.  All bets are off about whether
        this will actually work (ordering of bitfields is compiler
        dependent.'''
        if name == 'hw_available':
            return self.bitfield & 0x1 != 0
        elif name == 'wm_available':
            return self.bitfield & 0x2 != 0
        elif name == 'blit_hw':
            return self.bitfield & 0x200 != 0
        elif name == 'blit_hw_CC':
            return self.bitfield & 0x400 != 0
        elif name == 'blit_hw_A':
            return self.bitfield & 0x800 != 0
        elif name == 'blit_sw':
            return self.bitfield & 0x1000 != 0
        elif name == 'blit_sw_CC':
            return self.bitfield & 0x2000 != 0
        elif name == 'blit_sw_A':
            return self.bitfield & 0x4000 != 0
        elif name == 'blit_fill':
            return self.bitfield & 0x8000 != 0
        elif name == 'vfmt':    # Dereference vfmt pointer.
            if self._vfmt:
                return self._vfmt.contents
            return None
        
        # current_w and current_h added in SDL 1.2.10
        if name in ('current_w', 'current_h'):
            SDL.dll.assert_version_compatible(name, (1,2,10))
            return getattr(self, '_%s' % name)
        raise AttributeError, name


class SDL_Overlay(Structure):
    '''The YUV hardware video overlay.

    :Ivariables:
        `format` : int
            Overlay pixel layout.  One of SDL_YV12_OVERLAY, SDL_IYUV_OVERLAY,
            SDL_YUY2_OVERLAY, SDL_UYVY_OVERLAY, SDL_YVYU_OVERLAY.
        `w` : int
            Width of the overlay.
        `h` : int
            Height of the overlay.
        `planes` : int
            Number of planes of pixel data.
        `pitches` : list of int
            List of pitches for each plane.
        `pixels` : list of `SDL_array`
            List of pixel buffers, one for each plane.  All pixel buffers
            provided as byte buffers.

    '''
    _fields_ = [('format', c_uint),
                ('w', c_int),
                ('h', c_int),
                ('planes', c_int),
                ('pitches', POINTER(c_short)),
                ('_pixels', POINTER(POINTER(c_byte))),
                ('hwfuncs', c_void_p),
                ('hwdata', c_void_p),
                ('flags', c_uint)]

    def __getattr__(self, name):
        '''Retrieve bitfields as bool.  All bets are off about whether
        this will actually work (ordering of bitfields is compiler
        dependent.'''
        if name == 'hw_overlay':
            return self.flags & 0x1 != 0
        
        elif name == 'pixels':
            if not self._pixels:
                raise SDL.error.SDL_Exception, 'Overlay needs locking'
            p = []
            for i in range(self.planes):
                sz = self.pitches[i] * self.h
                p.append(SDL.array.SDL_array(self._pixels[i], sz, c_byte))
            return p


# SDL_VideoInit and SDL_VideoQuit not implemented (internal only, according
# to SDL_video.h).

_SDL_VideoDriverName = SDL.dll.private_function('SDL_VideoDriverName',
    arg_types=[c_char_p, c_int],
    return_type=c_int)

def SDL_VideoDriverName(maxlen=1024):
    '''
    Returns the name of the video driver.

    :Parameters:
        `maxlen`
            Maximum length of the returned driver name; defaults to 1024.

    :rtype: string
    '''
    buf = create_string_buffer(maxlen)
    if _SDL_VideoDriverName(buf, maxlen):
        return buf.value
    return None

SDL_GetVideoSurface = SDL.dll.function('SDL_GetVideoSurface',
    '''Return the current display surface.

    If SDL is doing format conversion on the display surface, this
    function returns the publicly visible surface, not the real
    video surface.

    :rtype: SDL_Surface
    ''',
    return_type=POINTER(SDL_Surface),
    dereference_return=True)

SDL_GetVideoInfo = SDL.dll.function('SDL_GetVideoInfo',
    '''Return information about the video hardware.

    If this is called before `SDL_SetVideoMode`, the ``vfmt`` member
    of the returned structure will contain the pixel format of the
    "best" video mode.

    :rtype: SDL_VideoInfo
    ''',
    return_type=POINTER(SDL_VideoInfo),
    dereference_return=True)

SDL_VideoModeOK = SDL.dll.function('SDL_VideoModeOK',
    '''Check to see if a particular video mode is supported.

    Returns 0 if the requested mode is not supported under any bit
    depth, or returns the bits-per-pixel of the closest available
    mode with the given width and height.  If this bits-per-pixel
    is different from the one used when setting the video mode,
    `SDL_SetVideoMode` will succeed, but will emulate the requested
    bits-per-pixel with a shadow surface.

    The arguments to `SDL_VideoModeOK` are the same ones you would
    pass to `SDL_SetVideoMode`.

    :Parameters:
      - `width`: int
      - `height`: int
      - `bpp`: int
      - `flags`: int

    :rtype: int

    :return: bits-per-pixel of the closest available mode, or 0 if 
        the mode is not supported.
    ''',
    args=['width', 'height', 'bpp', 'flags'],
    arg_types=[c_int, c_int, c_int, c_uint],
    return_type=c_int)

_SDL_ListModes = SDL.dll.private_function('SDL_ListModes',
    arg_types=[POINTER(SDL_PixelFormat), c_uint],
    return_type=POINTER(POINTER(SDL_Rect)))

def SDL_ListModes(format, flags):
    '''Return a list of available screen dimensions for the given
    format and video flags, sorted largest to smallest.

    Returns -1 if any size can be used, otherwise a list of `SDL_Rect`
    (which may be empty).

    If `format` is ``None``, the mode list will be for the format
    given by ``SDL_GetVideoInfo().vfmt``.

    :Parameters:
      - `format`: `SDL_PixelFormat`
      - `flags`: int

    :rtype: list
    :return: -1 or list of `SDL_Rect`
    '''
    ar = _SDL_ListModes(format, flags)
    if not ar:
        return []
    if addressof(ar.contents) == -1:
        return SDL_ANY_DIMENSION
    i = 0
    lst = []
    while ar[i]:
        lst.append(ar[i].contents)
        i += 1
    return lst

SDL_SetVideoMode = SDL.dll.function('SDL_SetVideoMode',
    '''Set up a video mode with the specified width, height and
    bits-per-pixel.

    If `bpp` is 0, it is treated as the current display bits-per-pixel.

    The `flags` can be a bitwise-OR of:

        `SDL_ANYFORMAT` 
            the SDL library will try to set the requested
            bits-per-pixel, but will return whatever video pixel format
            is available.  The default is to emulate the requested pixel
            format if it is not natively available.
        `SDL_HWSURFACE` 
            the video surface will be placed in video memory, if
            possible, and you may have to call `SDL_LockSurface` in
            order to access the raw framebuffer.  Otherwise, the video
            surface will be created in system memory.
        `SDL_ASYNCBLIT` 
            SDL will try to perform rectangle updates asynchronously,
            but you must always lock before accessing pixels.  SDL will
            wait for updates to complete before returning from the lock.
        `SDL_HWPALETTE` 
            the SDL library will guarantee that the colors set by
            `SDL_SetColors` will be the colors you get.  Otherwise, in
            8-bit mode, `SDL_SetColors` may not be able to set all of
            the colors exactly the way they are requested, and you
            should look at the video surface structure to determine the
            actual palette.  If SDL cannot guarantee that the colors you
            request can be set, i.e.  if the colormap is shared, then
            the video surface may be created under emulation in system
            memory, overriding the `SDL_HWSURFACE` flag.
        `SDL_FULLSCREEN` 
            the SDL library will try to set a fullscreen video mode.
            The default is to create a windowed mode if the current
            graphics system has a window manager.  If the SDL library is
            able to set a fullscreen video mode, this flag will be set
            in the surface that is returned.
        `SDL_DOUBLEBUF`
            the SDL library will try to set up two surfaces in video
            memory and swap between them when you call `SDL_Flip`.  This
            is usually slower than the normal single-buffering scheme,
            but prevents "tearing" artifacts caused by modifying video
            memory while the monitor is refreshing.  It should only be
            used by applications that redraw the entire screen on every
            update.
        `SDL_RESIZABLE` 
            the SDL library will allow the window manager, if any, to
            resize the window at runtime.  When this occurs, SDL will
            send a `SDL_VIDEORESIZE` event to you application, and you
            must respond to the event by re-calling `SDL_SetVideoMode`
            with the requested size (or another size that suits the
            application).
        `SDL_NOFRAME` 
            the SDL library will create a window without any title bar
            or frame decoration.  Fullscreen video modes have this flag
            set automatically.

    This function returns the video framebuffer surface, or ``None`` if
    it fails.

    If you rely on functionality provided by certain video flags, check
    the flags of the returned surface to make sure that functionality is
    available.  SDL will fall back to reduced functionality if the exact
    flags you wanted are not available.

    :Parameters:
      - `width`: int
      - `height`: int
      - `bpp`: int
      - `flags`: int

    :rtype: `SDL_Surface`
    ''',
    args=['width', 'height', 'bpp', 'flags'],
    arg_types=[c_int, c_int, c_int, c_uint],
    return_type=POINTER(SDL_Surface),
    dereference_return=True,
    require_return=True)

_SDL_UpdateRects = SDL.dll.private_function('SDL_UpdateRects',
    arg_types=[POINTER(SDL_Surface), c_int, POINTER(SDL_Rect)],
    return_type=None)

def SDL_UpdateRects(screen, rects):
    '''Make sure that the given list of rectangles is updated on the
    given screen.

    This function should not be called while `screen` is locked.

    :Parameters:
      - `screen`: `SDL_Surface`
      - `rects`: list of `SDL_Rect`
    '''
    ref, ar = SDL.array.to_ctypes(rects, len(rects), SDL_Rect)
    _SDL_UpdateRects(screen, len(rects), ar)

SDL_UpdateRect = SDL.dll.function('SDL_UpdateRect',
    '''Make sure that the given rectangle is updated on the given
    screen.

    If `x`, `y`, `w` and `h` are all 0, `SDL_UpdateRect` will update the
    entire screen.

    This function should not be called while `screen` is locked.

    :Parameters:
      - `screen`: `SDL_Surface`
      - `x`: int
      - `y`: int
      - `w`: int
      - `h`: int
    ''',
    args=['screen', 'x', 'y', 'w', 'h'],
    arg_types=[POINTER(SDL_Surface), c_int, c_int, c_uint, c_uint],
    return_type=None)

SDL_Flip = SDL.dll.function('SDL_Flip',
    '''Flip the front and back buffers.

    On hardware that supports double-buffering, this function sets up a
    flip and returns.  The hardware will wait for vertical retrace, and
    then swap video buffers before the next video surface blit or lock
    will return.

    On hardware that doesn not support double-buffering, this is
    equivalent to calling ``SDL_UpdateRect(screen, 0, 0, 0, 0)``.

    The `SDL_DOUBLEBUF` flag must have been passed to `SDL_SetVideoMode`
    when setting the video mode for this function to perform hardware
    flipping.

    :param `screen`: `SDL_Surface`
    ''',
    args=['screen'],
    arg_types=[POINTER(SDL_Surface)],
    return_type=c_int,
    success_return=0)

SDL_SetGamma = SDL.dll.function('SDL_SetGamma',
    '''Set the gamma correction for each of the color channels.

    The gamma values range (approximately) between 0.1 and 10.0
    
    If this function isn't supported directly by the hardware, it will
    be emulated using gamma ramps, if available. 

    :Parameters:
     - `red`: float
     - `green`: float
     - `blue`: float

    ''',
    args=['red', 'green', 'blue'],
    arg_types=[c_float, c_float, c_float],
    return_type=c_int,
    success_return=0)

_SDL_SetGammaRamp = SDL.dll.private_function('SDL_SetGammaRamp',
    arg_types=[POINTER(c_ushort), POINTER(c_ushort), POINTER(c_ushort)],
    return_type=c_int)

def SDL_SetGammaRamp(red, green, blue):
    '''Set the gamma translation table for the red, green and blue
    channels of the video hardware.

    Each table is a list of 256 ints in the range [0, 2**16),
    representing a mapping between the input and output for that
    channel.  The input is the index into the array, and the output is
    the 16-bit gamma value at that index, scaled to the output color
    precision.
    
    You may pass ``None`` for any of the channels to leave it unchanged.

    :Parameters:
        `red` : SDL_array
            For each channel you may pass either None (to leave the channel
            unchanged), an SDL_array or ctypes array of length 256, or
            any other sequence of length 256.
        `green` : SDL_array
            As above
        `blue` : SDL_array
            As above

    '''
    rar = gar = bar = None
    if red:
        rref, rar = SDL.array.to_ctypes(red, 256, c_ushort)
    if green:
        gref, gar = SDL.array.to_ctypes(green, 256, c_ushort)
    if blue:
        bref, bar = SDL.array.to_ctypes(blue, 256, c_ushort)
    result = _SDL_SetGammaRamp(rar, gar, bar)
    if result != 0:
        raise SDL.error.SDL_Exception, SDL.error.SDL_GetError()

_SDL_GetGammaRamp = SDL.dll.private_function('SDL_GetGammaRamp',
    arg_types=[POINTER(c_ushort), POINTER(c_ushort), POINTER(c_ushort)],
    return_type=c_int)

def SDL_GetGammaRamp():
    '''Retrieve the current values of the gamma translation tables.

    :rtype: (SDL_array, SDL_array, SDL_array)
    :return: a tuple (``red``, ``green``, ``blue``) where each element
        is either None (if the display driver doesn't support gamma
        translation) or an SDL_array of 256 ints in the range [0, 2**16).
    '''
    rar = SDL.array.SDL_array(None, 256, c_ushort)
    gar = SDL.array.SDL_array(None, 256, c_ushort)
    bar = SDL.array.SDL_array(None, 256, c_ushort)
    if _SDL_GetGammaRamp(rar.as_ctypes(), gar.as_ctypes(), bar.as_ctypes()) == 0:
        return rar, gar, bar
    return None, None, None

_SDL_SetColors = SDL.dll.private_function('SDL_SetColors',
    arg_types=[POINTER(SDL_Surface), POINTER(SDL_Color), c_int, c_int],
    return_type=c_int)

def SDL_SetColors(surface, colors, firstcolor):
    '''Sets a portion of the colormap for the given 8-bit surface.

    If `surface` is not a palettized surface, this function does
    nothing, returning 0.  If all of the colors were set as passed to
    `SDL_SetColors`, it will return 1.  If not all the color entries
    were set exactly as given, it will return 0, and you should look at
    the surface palette to determine the actual color palette.

    When `surface` is the surface associated with the current display, the
    display colormap will be updated with the requested colors.  If 
    `SDL_HWPALETTE` was set in `SDL_SetVideoMode` flags, `SDL_SetColors`
    will always return 1, and the palette is guaranteed to be set the way
    you desire, even if the window colormap has to be warped or run under
    emulation.
    
    :Parameters:
      - `surface`: `SDL_Surface`
      - `colors`: sequence or SDL_array of `SDL_Color`
      - `firstcolor`: int; the first color index to set.

    :rtype: int
    :return: 1 if all colors were set as passed, otherwise 0.
    '''
    ref, ar = SDL.array.to_ctypes(colors, len(colors), SDL_Color)
    return _SDL_SetColors(surface, ar, firstcolor, len(colors))

_SDL_SetPalette = SDL.dll.private_function('SDL_SetPalette',
    arg_types=[POINTER(SDL_Surface), c_int, POINTER(SDL_Color), c_int, c_int],
    return_type=c_int)

def SDL_SetPalette(surface, flags, colors, firstcolor):
    '''Sets a portion of the colormap for a given 8-bit surface.

    `flags` is one or both of:
        `SDL_LOGPAL`
            set logical palette, which controls how blits are mapped
            to/from the surface
        `SDL_PHYSPAL`
            set physical palette, which controls how pixels look on the
            screen.

    Only screens have physical palettes.  Separate change of
    physical/logical palettes is only possible if the screen has
    `SDL_HWPALETTE` set.

    `SDL_SetColors` is equivalent to calling this function with 
    ``flags = (SDL_LOGPAL | SDL_PHYSPAL)``.

    :Parameters:
      - `surface`: `SDL_Surface`
      - `flags`: int
      - `colors`: sequence or SDL_array of `SDL_Color`
      - `firstcolor`: int; the first color index to set.
    '''
    ref, ar = SDL.array.to_ctypes(colors, len(colors), SDL_Color)
    result = _SDL_SetPalette(surface, flags, ar, firstcolor, len(colors))
    if result != 1:
        raise SDL.error.SDL_Exception, SDL.error.SDL_GetError()

SDL_MapRGB = SDL.dll.function('SDL_MapRGB',
    '''Map an RGB triple to an opaque pixel value for a given pixel
    format.

    :Parameters:
     - `format`: `SDL_PixelFormat`
     - `r`: int in the range [0, 255]
     - `g`: int in the range [0, 255]
     - `b`: int in the range [0, 255]

    :rtype: int
    :return: the opaque pixel value
    ''',
    args=['format', 'r', 'g', 'b'],
    arg_types=[POINTER(SDL_PixelFormat), c_ubyte, c_ubyte, c_ubyte],
    return_type=c_uint)

SDL_MapRGBA = SDL.dll.function('SDL_MapRGBA',
    '''Map an RGBA quadruple to an opaque pixel value for a given pixel
    format.

    :Parameters:
     - `format`: `SDL_PixelFormat`
     - `r`: int in the range [0, 255]
     - `g`: int in the range [0, 255]
     - `b`: int in the range [0, 255]
     - `a`: int in the range [0, 255]

    :rtype: int
    :return: the opaque pixel value
    ''',
    args=['format', 'r', 'g', 'b', 'a'],
    arg_types=[POINTER(SDL_PixelFormat), c_ubyte, c_ubyte, c_ubyte, c_ubyte],
    return_type=c_uint)

_SDL_GetRGB = SDL.dll.private_function('SDL_GetRGB',
    arg_types=[c_uint, POINTER(SDL_PixelFormat), 
               POINTER(c_ubyte), POINTER(c_ubyte), POINTER(c_ubyte)],
    return_type=None)

def SDL_GetRGB(pixel, fmt):
    '''Map a pixel value into the RGB components for a given pixel
    format.

    :Parameters:
     - `pixel`: int
     - `fmt`: `SDL_PixelFormat`

    :rtype: (int, int, int)
    :return: tuple (``r``, ``g``, ``b``) where each element is an int
        in the range [0, 255]
    '''
    r, g, b = c_ubyte(), c_ubyte(), c_ubyte()
    _SDL_GetRGB(pixel, fmt, byref(r), byref(g), byref(b))
    return r.value, g.value, b.value

_SDL_GetRGBA = SDL.dll.private_function('SDL_GetRGBA',
    arg_types=[c_uint, POINTER(SDL_PixelFormat), 
               POINTER(c_ubyte), POINTER(c_ubyte), 
               POINTER(c_ubyte), POINTER(c_ubyte)],
    return_type=None)

def SDL_GetRGBA(pixel, fmt):
    '''Map a pixel value into the RGBA components for a given pixel
    format.

    :Parameters:
     - `pixel`: int
     - `fmt`: `SDL_PixelFormat`

    :rtype: (int, int, int, int)
    :return: tuple (``r``, ``g``, ``b``, ``a``) where each element is an
        int in the range [0, 255]
    '''
    r, g, b, a = c_ubyte(), c_ubyte(), c_ubyte(), c_ubyte()
    _SDL_GetRGBA(pixel, fmt, byref(r), byref(g), byref(b), byref(a))
    return r.value, g.value, b.value, a.value

SDL_CreateRGBSurface = SDL.dll.function('SDL_CreateRGBSurface',
    '''Allocate an RGB surface.
    
    Must be called after `SDL_SetVideoMode`.  If the depth is 4 or 8
    bits, an empty palette is allocated for the surface.  If the depth
    is greater than 8 bits, the pixel format is set using the flags
    ``[RGB]mask``. 

    The `flags` tell what kind of surface to create

        `SDL_SWSURFACE` 
            the surface should be created in system memory.
        `SDL_HWSURFACE` 
            the surface should be created in video memory, with the same
            format as the display surface.  This is useful for surfaces
            that will not change much, to take advantage of hardware
            acceleration when being blitted to the display surface.
        `SDL_ASYNCBLIT` 
            SDL will try to perform asynchronous blits with this
            surface, but you must always lock it before accessing the
            pixels.  SDL will wait for current blits to finish before
            returning from the lock.
        `SDL_SRCCOLORKEY` 
            the surface will be used for colorkey blits.  If the
            hardware supports acceleration of colorkey blits between two
            surfaces in video memory, SDL will try to place the surface
            in video memory. If this isn't possible or if there is no
            hardware acceleration available, the surface will be placed
            in system memory.
        `SDL_SRCALPHA` 
            the surface will be used for alpha blits and if the hardware
            supports hardware acceleration of alpha blits between two
            surfaces in video memory, to place the surface in video
            memory if possible, otherwise it will be placed in system
            memory.

    If the surface is created in video memory, blits will be _much_
    faster, but the surface format must be identical to the video
    surface format, and the only way to access the pixels member of the
    surface is to use the `SDL_LockSurface` and `SDL_UnlockSurface`
    calls.

    If the requested surface actually resides in video memory,
    `SDL_HWSURFACE` will be set in the flags member of the returned
    surface.  If for some reason the surface could not be placed in
    video memory, it will not have the `SDL_HWSURFACE` flag set, and
    will be created in system memory instead.

    :Parameters:
     - `flags`: int
     - `width`: int
     - `height`: int
     - `depth`: int
     - `Rmask`: int
     - `Gmask`: int
     - `Bmask`: int
     - `Amask`: int

    :rtype: `SDL_Surface`
    ''',
    args=['flags', 'width', 'height', 'depth', 'Rmask', 'Gmask',
          'Bmask', 'Amask'],
    arg_types=[c_uint, c_int, c_int, c_int, c_uint, c_uint, c_uint, c_uint],
    return_type=POINTER(SDL_Surface),
    dereference_return=True,
    require_return=True)

_SDL_CreateRGBSurfaceFrom = \
    SDL.dll.private_function('SDL_CreateRGBSurfaceFrom',
    arg_types=[POINTER(c_ubyte), c_int, c_int, c_int, 
               c_uint, c_uint, c_uint, c_uint],
    return_type=POINTER(SDL_Surface),
    dereference_return=True,
    require_return=True)

def SDL_CreateRGBSurfaceFrom(pixels, width, height, depth, pitch,
                             Rmask, Gmask, Bmask, Amask):
    '''Allocate and initialise an RGB surface.

    Must be called after `SDL_SetVideoMode`.  If the depth is 4 or 8
    bits, an empty palette is allocated for the surface.  If the depth
    is greater than 8 bits, the pixel format is set using the flags
    ``[RGB]mask``. 

    `pixels` may be any sequence of integers of size `depth`, or a
    sequence of bytes, an SDL_array or a numpy array.  The type
    of data passed will be detected based on the values of `width`,
    `height` and `depth`.  If `depth` is less than 8 bits, you must
    pass the data as bytes.

    :Parameters:
     - `pixels`: sequence or SDL_array
     - `width`: int
     - `height`: int
     - `depth`: int
     - `pitch`: int
     - `Rmask`: int
     - `Gmask`: int
     - `Bmask`: int
     - `Amask`: int

    :rtype: `SDL_Surface`
    '''
    if hasattr(pixels, 'contents'):
        # accept ctypes pointer without modification
        ref, ar = None, pixels
    else:
        len_pixels = len(pixels)
        if len(pixels) == pitch * 8 / depth * height:
            # pixel array?
            if depth == 8:
                ref, ar = SDL.array.to_ctypes(pixels, len(pixels), c_ubyte)
            elif depth == 16:
                ref, ar = SDL.array.to_ctypes(pixels, len(pixels), c_ushort)
            elif depth == 32:
                ref, ar = SDL.array.to_ctypes(pixels, len(pixels), c_uint)
        elif len(pixels) == pitch * height:
            # byte array
            ref, ar = SDL.array.to_ctypes(pixels, len(pixels), c_ubyte)
        else:
            raise TypeError, 'Length of pixels does not match given dimensions.'

    surface = _SDL_CreateRGBSurfaceFrom(cast(ar, POINTER(c_ubyte)),
        width, height, depth, pitch, Rmask, Gmask, Bmask, Amask)
    surface._buffer_ref = ref
    return surface

SDL_FreeSurface = SDL.dll.function('SDL_FreeSurface',
    '''Free an RGB Surface.

    :Parameters:
     - `surface`: `SDL_Surface`
    ''',
    args=['surface'],
    arg_types=[POINTER(SDL_Surface)],
    return_type=None)

SDL_LockSurface = SDL.dll.function('SDL_LockSurface',
    '''Set up a surface for directly accessing the pixels.

    Between calls to `SDL_LockSurface`/`SDL_UnlockSurface`, you can write
    to and read from ``surface.pixels``, using the pixel format stored in 
    ``surface.format``.  Once you are done accessing the surface, you should 
    use `SDL_UnlockSurface` to release it.

    Not all surfaces require locking.  If `SDL_MUSTLOCK` evaluates
    to 0, then you can read and write to the surface at any time, and the
    pixel format of the surface will not change.  In particular, if the
    `SDL_HWSURFACE` flag is not given when calling `SDL_SetVideoMode`, you
    will not need to lock the display surface before accessing it.
    
    No operating system or library calls should be made between lock/unlock
    pairs, as critical system locks may be held during this time.

    :Parameters:
     - `surface`: `SDL_Surface`
    ''',
    args=['surface'],
    arg_types=[POINTER(SDL_Surface)],
    return_type=c_int,
    success_return=0)

SDL_UnlockSurface = SDL.dll.function('SDL_UnlockSurface',
    '''Unlock a surface locked with `SDL_LockSurface`.

    :Parameters:
     - `surface`: `SDL_Surface`
    ''',
    args=['surface'],
    arg_types=[POINTER(SDL_Surface)],
    return_type=None)

SDL_LoadBMP_RW = SDL.dll.function('SDL_LoadBMP_RW',
    '''Load a surface from a seekable SDL data source (memory or file).

    If `freesrc` is non-zero, the source will be closed after being read.
    The new surface which is returned should be freed with
    `SDL_FreeSurface`.

    :Parameters:
     - `src`: `SDL_RWops`
     - `freesrc`: int

    :rtype: `SDL_Surface`
    ''',
    args=['src', 'freesrc'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops), c_int],
    return_type=POINTER(SDL_Surface),
    dereference_return=True,
    require_return=True)

def SDL_LoadBMP(file):
    '''Load a surface from a file.

    Convenience function equivalent to
    ``SDL_LoadBMP_RW(SDL_RWFromFile(file, 'rb'), 1)``.

    :Parameters:
     - `file`: string, the location of the bitmap file.

    :rtype: `SDL_Surface`
    '''
    return SDL_LoadBMP_RW(SDL.rwops.SDL_RWFromFile(file, 'rb'), 1)

SDL_SaveBMP_RW = SDL.dll.function('SDL_SaveBMP_RW',
    '''Save a surface to a seekable SDL data source (memory or file).

    If `freedst` is non-zero, the destination will be closed after being
    written.  

    :Parameters:
     - `surface`: `SDL_Surface`
     - `dst`: `SDL_RWops`
     - `freesrc`: int
    ''',
    args=['surface', 'dst', 'freedst'],
    arg_types=[POINTER(SDL_Surface), POINTER(SDL.rwops.SDL_RWops), c_int],
    return_type=c_int,
    success_return=0)

def SDL_SaveBMP(surface, file):
    '''Save a surface to a file.

    Convenience function equivalent to
    ``SDL_SaveBMP_RW(surface, SDL_RWFromFile(file, 'wb'), 1)``

    :Parameters:
     - `surface`: `SDL_Surface`
     - `dst`: `SDL_RWops`
    '''
    return SDL_SaveBMP_RW(surface, SDL.rwops.SDL_RWFromFile(file, 'wb'), 1)

SDL_SetColorKey = SDL.dll.function('SDL_SetColorKey',
    '''Set the color key (transparent pixel) in a blittable surface.

    If `flag` is `SDL_SRCCOLORKEY` (optionally OR'd with `SDL_RLEACCEL`), 
    `key` will be the transparent pixel in the source image of a blit.
    `SDL_RLEACCEL` requests RLE acceleration for the surface if present,
    and removes RLE acceleration if absent.  If `flag` is 0, this
    function clears any current color key.

    :Parameters:
     - `surface`: `SDL_Surface`
     - `flag`: int
     - `key`: int, see `SDL_MapRGB`
    ''',
    args=['surface', 'flag', 'key'],
    arg_types=[POINTER(SDL_Surface), c_uint, c_uint],
    return_type=c_int,
    success_return=0)

SDL_SetAlpha = SDL.dll.function('SDL_SetAlpha',
    '''Set the alpha value for the entire surface, as opposed to using
    the alpha component of each pixel.

    This value measures the range of transparency of the surface, 0
    being completely transparent to 255 being completely opaque. An
    `alpha` value of 255 causes blits to be opaque, the source pixels
    copied to the destination (the default). Note that per-surface alpha
    can be combined with colorkey transparency.

    If `flag` is 0, alpha blending is disabled for the surface.  If
    `flag` is `SDL_SRCALPHA`, alpha blending is enabled for the surface.
    OR'ing the flag with `SDL_RLEACCEL` requests RLE acceleration for
    the surface; if `SDL_RLEACCEL` is not specified, the RLE accel will
    be removed.

    The `alpha` parameter is ignored for surfaces that have an alpha
    channel.

    :Parameters:
     - `surface`: `SDL_Surface`
     - `flag`: int
     - `alpha`: int

    :return: int; undocumented. (FIXME)
    ''',
    args=['surface', 'flag', 'alpha'],
    arg_types=[POINTER(SDL_Surface), c_uint, c_uint],
    return_type=c_int)

SDL_SetClipRect = SDL.dll.function('SDL_SetClipRect',
    '''Set the clipping rectangle for the destination surface in a blit.

    If the clip rectangle is None, clipping will be disabled.
    If the clip rectangle doesn't intersect the surface, the function will
    return False and blits will be completely clipped.  Otherwise the
    function returns True and blits to the surface will be clipped to
    the intersection of the surface area and the clipping rectangle.

    Note that blits are automatically clipped to the edges of the source
    and destination surfaces.

    :Parameters:
     - `surface`: `SDL_Surface`
     - `rect`: `SDL_Rect` or None

    :rtype: int
    :return: non-zero if blitting will not be entirely clipped, otherwise
             zero.
    ''',
    args=['surface', 'rect'],
    arg_types=[POINTER(SDL_Surface), POINTER(SDL_Rect)],
    return_type=c_int)

_SDL_GetClipRect = SDL.dll.private_function('SDL_GetClipRect',
    arg_types=[POINTER(SDL_Surface), POINTER(SDL_Rect)],
    return_type=None)

def SDL_GetClipRect(surface):
    '''Get the clipping rectangle for the destination surface in a blit.

    :see: `SDL_SetClipRect`

    :Parameters:
      - `surface`: `SDL_Surface`

    :rtype: `SDL_Rect`
    '''
    rect = SDL_Rect()
    _SDL_GetClipRect(surface, byref(rect))
    return rect

SDL_ConvertSurface = SDL.dll.function('SDL_ConvertSurface',
    '''Create a new surface of the specified format, then copy and
    map the given surface to it so the blit of the converted surface
    will be as fast as possible.

    The `flags` parameter is passed to `SDL_CreateRGBSurface` and has those 
    semantics.  You can also pass `SDL_RLEACCEL` in the flags parameter and
    SDL will try to RLE accelerate colorkey and alpha blits in the resulting
    surface.

    This function is used internally by `SDL_DisplayFormat`.

    :Parameters:
     - `src`: `SDL_Surface`
     - `fmt`: `SDL_PixelFormat`
     - `flags`: int

    :rtype: `SDL_Surface`
    ''',
    args=['src', 'fmt', 'flags'],
    arg_types=[POINTER(SDL_Surface), POINTER(SDL_PixelFormat), c_uint],
    return_type=POINTER(SDL_Surface),
    dereference_return=True,
    require_return=True)

SDL_UpperBlit = SDL.dll.function('SDL_UpperBlit',
    '''Perform a fast blit from the source surface to the destination
    surface.

    ``SDL_BlitSurface`` is a synonym for ``SDL_UpperBlit``.

    This function assumes that the source and destination rectangles are
    the same size.  If either `srcrect` or `dstrect` are None, the entire
    surface (`src` or `dst`) is copied.  The final blit rectangles are saved
    in `srcrect` and `dstrect` after all clipping is performed.

    The blit function should not be called on a locked surface.

    The blit semantics for surfaces with and without alpha and colorkey
    are defined as follows:

        RGBA to RGB
            `SDL_SRCALPHA` set:
                - alpha-blend (using alpha-channel)
                - `SDL_SRCCOLORKEY` ignored
            `SDL_SRCALPHA` not set:
                - copy RGB
                - if `SDL_SRCCOLORKEY` set, only copy the pixels
                  matching the RGB value of the source color key,
                  ignoring alpha in the comparison.
        RGB to RGBA:
            `SDL_SRCALPHA` set:
                - alpha-blend (using the source per-surface alpha value)
                - set destination alpha to opaque
            `SDL_SRCALPHA` not set:
                - copy RGB, set destination alpha to source per-surface
                  alpha value.
            `SDL_SRCCOLORKEY` set:
                - only copy pixels matching the source color key
        RGBA to RGBA:
            `SDL_SRCALPHA` set:
                - alpha-blend (using the source alpha channel) the RGB
                  values; leave destination alpha untouched.
                - `SDL_SRCCOLORKEY` ignored.
            `SDL_SRCALPHA` not set:
                - copy all of RGBA to the destination
                - if `SDL_SRCCOLORKEY` set, only copy the pixels
                  matching the RGB value of the source color key,
                  ignoring alpha in the comparison.
        RGB to RGB:
            `SDL_SRCHALPHA` set:
                - alpha-blend (using the source per-surface alpha value)
            `SDL_SRCALPHA` not set:
                - copy RGB
            `SDL_SRCCOLORKEY` set:
                - only copy the pixels matching the source color key.

    If either of the surfaces were in video memory, and the blit returns
    -2, the video memory was lost, so it should be reloaded with artwork
    and reblitted::

        while SDL_BlitSurface(image, imgrect, screen, dstrect) == -2:
            while SDL_LockSurface(image) < 0:
                sleep(10)
            # Write image pixels to image.pixels
            SDL_UnlockSurface(image)

    This happens under DirectX 5.0 when the system switches away from
    your fullscreen application.  The lock will also fail until you have
    access to the video memory again.

    :Parameters:
     - `src`: `SDL_Surface`
     - `srcrect`: `SDL_Rect`
     - `dst`: `SDL_Surface`
     - `dstrect`: `SDL_Rect`

    :rtype: int
    :return: 0 if successful, or -2 if there was an error and video 
             memory was lost.  (any other error will raise an exception)
    ''',
    args=['src', 'srcrect', 'dst', 'dstrect'],
    arg_types=[POINTER(SDL_Surface), POINTER(SDL_Rect),
               POINTER(SDL_Surface), POINTER(SDL_Rect)],
    return_type=c_int,
    error_return=-1)

SDL_BlitSurface = SDL.dll.function('SDL_UpperBlit',
    '''Perform a fast blit from the source surface to the destination
    surface.

    This function is a synonym for `SDL_UpperBlit`.

    :rtype: int
    ''',
    args=['src', 'srcrect', 'dst', 'dstrect'],
    arg_types=[POINTER(SDL_Surface), POINTER(SDL_Rect),
               POINTER(SDL_Surface), POINTER(SDL_Rect)],
    return_type=c_int,
    error_return=-1)

SDL_LowerBlit = SDL.dll.function('SDL_LowerBlit',
    '''Low-level fast blit.

    This is a semi-private blit function that does not perform
    rectangle validation or clipping.  See `SDL_UpperBlit` for the
    public function.

    :rtype: int
    ''',
    args=['src', 'srcrect', 'dst', 'dstrect'],
    arg_types=[POINTER(SDL_Surface), POINTER(SDL_Rect),
               POINTER(SDL_Surface), POINTER(SDL_Rect)],
    return_type=c_int,
    error_return=-1)

SDL_FillRect = SDL.dll.function('SDL_FillRect',
    '''Perform a fast fill of the given rectangle with `color`.

    The given rectangle is clipped to the destination surface clip area
    and the final fill rectangle is saved in the passed in rectangle.
    If `dstrect` is None, the whole surface will be filled with `color`
    The color should be a pixel of the format used by the surface, and 
    can be generated by the `SDL_MapRGB` function.

    :Parameters:
     - `dst`: `SDL_Surface`
     - `dstrect`: `SDL_Rect`
     - `color`: int
    ''',
    args=['dst', 'dstrect', 'color'],
    arg_types=[POINTER(SDL_Surface), POINTER(SDL_Rect), c_uint],
    return_type=c_int,
    success_return=0)

SDL_DisplayFormat = SDL.dll.function('SDL_DisplayFormat',
    '''Copy a surface to the pixel format and colors of the display
    buffer.

    This function makes a copy of a surface suitable for fast blitting
    onto the display surface.  It calls `SDL_ConvertSurface`.
    
    If you want to take advantage of hardware colorkey or alpha blit
    acceleration, you should set the colorkey and alpha value before
    calling this function.

    :Parameters:
     - `surface`: `SDL_Surface`

    :rtype: `SDL_Surface`
    ''',
    args=['surface'],
    arg_types=[POINTER(SDL_Surface)],
    return_type=POINTER(SDL_Surface),
    dereference_return=True,
    require_return=True)

SDL_DisplayFormatAlpha = SDL.dll.function('SDL_DisplayFormatAlpha',
    '''Copy a surface to the pixel format and colors of the display
    buffer.

    This function makes a copy of a surface suitable for fast alpha
    blitting onto the display surface.  The new surface will always have
    an alpha channel.
    
    If you want to take advantage of hardware colorkey or alpha blit
    acceleration, you should set the colorkey and alpha value before
    calling this function.

    :Parameters:
     - `surface`: `SDL_Surface`

    :rtype: `SDL_Surface`
    ''',
    args=['surface'],
    arg_types=[POINTER(SDL_Surface)],
    return_type=POINTER(SDL_Surface),
    dereference_return=True,
    require_return=True)

SDL_CreateYUVOverlay = SDL.dll.function('SDL_CreateYUVOverlay',
    '''Create a video output overlay.

    Calling the returned surface an overlay is something of a misnomer
    because the contents of the display surface underneath the area
    where the ovelay is shown is undefined - it may be overwritten with
    the converted YUV data.

    The most common video overlay formats are:
        
        `SDL_YV12_OVERLAY`
            Planar mode: Y + V + U (3 planes)
        `SDL_IYUV_OVERLAY`
            Planar mode: Y + U + V (3 planes)
        `SDL_YUY2_OVERLAY`
            Packed moed: Y0 + U0 + Y1 + V0 (1 plane)
        `SDL_UYVY_OVERLAY`
            Packed moed: U0 + Y0 + V0 + Y1 (1 plane)
        `SDL_YVYU_OVERLAY`
            Packed moed: Y0 + V0 + Y1 + U0 (1 plane)

    For an explanation of these pixel formats, see
    http://www.webartz.com/fourcc/indexyuv.htm

    For more information on the relationship between color spaces, see:
    http://www.neuro.sfc.keio.ac.jp/~aly/polygon/info/color-space-faq.html

    :Parameters:
     - `width`: int
     - `height`: int
     - `format`: int
     - `display`: `SDL_Surface`

    :rtype: `SDL_Overlay`
    ''',
    args=['width', 'height', 'format', 'display'],
    arg_types=[c_int, c_int, c_uint, POINTER(SDL_Surface)],
    return_type=POINTER(SDL_Overlay),
    dereference_return=True,
    require_return=True)

SDL_LockYUVOverlay = SDL.dll.function('SDL_LockYUVOverlay',
    '''Lock an overlay for direct access.

    Unlock the overlay when done with `SDL_UnlockYUVOverlay`.

    :Parameters:
     - `overlay`: `SDL_Overlay`

    :rtype: int
    :return: undocumented (FIXME)
    ''',
    args=['overlay'],
    arg_types=[POINTER(SDL_Overlay)],
    return_type=c_int)

SDL_UnlockYUVOverlay = SDL.dll.function('SDL_UnlockYUVOverlay',
    '''Unlock an overlay after locking it with `SDL_LockYUVOverlay`.

    :Parameters:
     - `overlay`: `SDL_Overlay`
    ''',
    args=['overlay'],
    arg_types=[POINTER(SDL_Overlay)],
    return_type=None)

SDL_DisplayYUVOverlay = SDL.dll.function('SDL_DisplayYUVOverlay',
    '''Blit a video overlay to the display surface.

    The contents of the video surface underneath the blit destination
    are not defined.  The width and height of the destination rectangle
    may be different from that of the overlay, but currently only 2x
    scaling is supported.

    :Parameters:
     - `overlay`: `SDL_Overlay`
     - `dstrect`: `SDL_Rect`

    :rtype: int
    :return: undocumented (FIXME)
    ''',
    args=['overlay', 'dstrect'],
    arg_types=[POINTER(SDL_Overlay), POINTER(SDL_Rect)],
    return_type=c_int)

SDL_FreeYUVOverlay = SDL.dll.function('SDL_FreeYUVOverlay',
    '''Free a video overlay.

    :Parameters:
     - `overlay`: `SDL_Overlay`

    ''',
    args=['overlay'],
    arg_types=[POINTER(SDL_Overlay)],
    return_type=None)

# SDL_GL_LoadLibrary, SDL_GL_GetProcAddress not implemented.

SDL_GL_SetAttribute = SDL.dll.function('SDL_GL_SetAttribute',
    '''Set an attribute of the OpenGL subsystem before initialization.

    :Parameters:
     - `attr`: int
     - `value`: int

    :rtype: int
    :return: undocumented (FIXME)
    ''',
    args=['attr', 'value'],
    arg_types=[c_uint, c_int],
    return_type=c_int)

_SDL_GL_GetAttribute = SDL.dll.private_function('SDL_GL_GetAttribute',
    arg_types=[c_int, POINTER(c_int)],
    return_type=c_int)

def SDL_GL_GetAttribute(attr):
    '''Get an attribute of the OpenGL subsystem from the windowing
    interface, such as glX.
    '''
    val = c_int()
    _SDL_GL_GetAttribute(attr, byref(val))
    return val.value

SDL_GL_SwapBuffers = SDL.dll.function('SDL_GL_SwapBuffers',
    '''Swap the OpenGL buffers, if double-buffering is supported.
    ''',
    args=[],
    arg_types=[],
    return_type=None)

# SDL_GL_UpdateRects, SDL_GL_Lock and SDL_GL_Unlock not implemented (marked
# private in SDL_video.h)

_SDL_WM_SetCaption = SDL.dll.private_function('SDL_WM_SetCaption',
    arg_types=[c_char_p, c_char_p],
    return_type=None)

def SDL_WM_SetCaption(title, icon):
    '''Set the title and icon text of the display window.

    Unicode strings are also accepted since 1.2.10.

    :Parameters:
     - `title`: string
     - `icon`: string
    '''
    _SDL_WM_SetCaption(title.encode('utf-8'), icon.encode('utf-8'))


_SDL_WM_GetCaption = SDL.dll.private_function('SDL_WM_GetCaption',
    arg_types=[POINTER(c_char_p), POINTER(c_char_p)],
    return_type=None)

def SDL_WM_GetCaption():
    '''Get the title and icon text of the display window.

    :rtype: (string, string)
    :return: a tuple (title, icon) where each element is a Unicode string
    '''
    title, icon = c_char_p(), c_char_p()
    _SDL_WM_GetCaption(byref(title), byref(icon))

    if title.value:
        title = title.value.decode('utf-8')
    else:
        title = None

    if icon.value:
        icon = icon.value.decode('utf-8')
    else:
        icon = None
    return title, icon

_SDL_WM_SetIcon = SDL.dll.private_function('SDL_WM_SetIcon',
    arg_types=[POINTER(SDL_Surface), POINTER(c_ubyte)],
    return_type=None)

def SDL_WM_SetIcon(icon, mask):
    '''Set the icon for the display window.

    This function must be called before the first call to `SDL_SetVideoMode`.
    It takes an icon surface, and a mask in MSB format.  If `mask` is None,
    the entire icon surface will be used as the icon.

    :Parameters:
     - `icon`: `SDL_Surface`
     - `mask`: `SDL_array` or sequence 

    '''
    if mask:
        ref, mask = \
            SDL.array.to_ctypes(mask, (icon.w * icon.h + 7) / 8, c_ubyte)
    _SDL_WM_SetIcon(icon, mask)

SDL_WM_IconifyWindow = SDL.dll.function('SDL_WM_IconifyWindow',
    '''Iconify the window.

    If the function succeeds, it generates an `SDL_APPACTIVATE` loss
    event.
    ''',
    args=[],
    arg_types=[],
    return_type=c_int,
    error_return=0)

SDL_WM_ToggleFullScreen = SDL.dll.function('SDL_WM_ToggleFullScreen',
    '''Toggle fullscreen mode without changing the contents of the
    screen.

    If the display surface does not require locking before accessing
    the pixel information, then the memory pointers will not change.

    The next call to `SDL_SetVideoMode` will set the mode fullscreen
    attribute based on the flags parameter - if `SDL_FULLSCREEN` is not
    set, then the display will be windowed by default where supported.

    This is currently only implemented in the X11 video driver.

    :Parameters:
     - `surface`: `SDL_Surface`
    ''',
    args=['surface'],
    arg_types=[POINTER(SDL_Surface)],
    return_type=c_int,
    error_return=0)

SDL_WM_GrabInput = SDL.dll.function('SDL_WM_GrabInput',
    '''Set the grab mode for the mouse and keyboard.

    Grabbing means that the mouse is confined to the application window,
    and nearly all keyboard input is passed directly to the application,
    and not interpreted by a window manager, if any.

    The behaviour of this function depends on the value of `mode`:

        `SDL_GRAB_QUERY`
            Return the current input grab mode.
        `SDL_GRAB_OFF`
            Turn off grab mode and return the new input grab mode.
        `SDL_GRAB_ON`
            Turn on grab mode and return the new input grab mode.

    :Parameters:
      - `mode`: int
            
    :rtype: int
    :return: the new input grab mode.
    ''',
    args=['mode'],
    arg_types=[c_int],
    return_type=c_int)

# SDL_SoftStretch not implemented (marked private in SDL_video.h)
