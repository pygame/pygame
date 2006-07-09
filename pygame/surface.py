#!/usr/bin/env python

'''Surface module containing the Surface class.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from copy import copy

from SDL import *
import pygame.base
import pygame.rect

class _SubSurface_Data(object):
    __slots__ = ['owner', 'pixeloffset', 'offsetx', 'offsety']

class Surface(object):
    __slots__ = ['_surf', '_subsurface']

    def __init__(self, size=(0,0), flags=0, depth=0, masks=None, surf=None):
        '''Create a new surface.

        Creates a new surface object.   `depth` and `masks` can be substituted
        for another `Surface` object which will create the new surface with
        the same format as the given one. 
        
        When using default masks, alpha will always be ignored unless you pass
        SRCALPHA as a flag.

        For a plain software surface, 0 can be used for the flag.  A plain
        hardware surface can just use 1 for the flag.

        :Parameters:
            `size` : int, int
                width, height
            `flags` : int
                A mix of the following flags: SWSURFACE, HWSURFACE, ASYNCBLIT,
                or SRCALPHA. (flags = 0 is the same as SWSURFACE).
            `depth` : int
                The number of bits used per pixel. If omitted, depth will use
                the current display depth.
            `masks` : int, int, int, int
                The bitmask for r,g,b, and a. If omitted, masks will default
                to the usual values for the given bitdepth.
            `surf` : `SDL_Surface`
                If specified, all other parameters are ignored.  The pygame
                surface will wrap the given `SDL_Surface` (used internally).

        '''
        if surf:
            if not isinstance(surf, SDL_Surface):
                raise TypeError, 'surf'
            self._surf = surf
            self._subsurface = None
        else:
            width, height = size
            if width < 0 or height < 0:
                raise pygame.base.error, 'Invalid resolution for Surface'

            # __init__ can be called more than once
            self._cleanup()

            format = None
            if not masks or not depth:
                if isinstance(depth, Surface):
                    format = depth._surf.format
                    depth = None
                elif isinstance(masks, Surface):
                    format = masks._surf.format
                    masks = None
                elif SDL_GetVideoSurface():
                    format = SDL_GetVideoSurface().format
                elif SDL_WasInit(SDL_INIT_VIDEO):
                    format = SDL_GetVideoInfo().vfmt 

            if not depth and not masks:
                if format:
                    masks = [format.Rmask, format.Gmask, format.Bmask, 0]

            if not depth:
                if format:
                    depth = format.BitsPerPixel
                else:
                    depth = 32

            if not masks:
                if flags & SDL_SRCALPHA:
                    if depth == 16:
                        masks = (0xf << 8, 0xf << 4, 0xf, 0xf << 12)
                    elif depth == 32:
                        masks = (0xff << 16, 0xff << 8, 0xff, 0xff << 24)
                    else:
                        raise ValueError, \
                          'no standard masks exist for given depth with alpha'
                else:
                    if depth == 8:
                        masks = (0xff >> 6 << 5, 0xff >> 5 << 2, 0xff >> 6, 0)
                    elif depth == 12:
                        masks = (0xff >> 4 << 8, 0xff >> 4 << 4, 0xff >> 4, 0)
                    elif depth == 15:
                        masks = (0xff >> 3 << 10, 0xff >> 3 << 5, 0xff >> 3, 0)
                    elif depth == 16:
                        masks = (0xff >> 3 << 11, 0xff >> 2 << 5, 0xff >> 3, 0)
                    elif depth == 24 or depth == 32:
                        masks = (0xff << 16, 0xff << 8, 0xff, 0)
                    else:
                        raise ValueError, 'nonstandard bit depth given'

            self._surf = SDL_CreateRGBSurface(flags, width, height, depth,
                masks[0], masks[1], masks[2], masks[3])

    def _cleanup(self):
        if hasattr(self, '_surf') and self._surf:
            if not (self._surf.flags & SDL_HWSURFACE) or \
               SDL_WasInit(SDL_INIT_VIDEO):
                # Unsafe to free hardware surface without video init
                SDL_FreeSurface(self._surf)
            self._surf = None
        self._subsurface = None

    def _prep(self):
        data = self._subsurface
        if data:
            data.owner.lock()
            self._surf._pixels = \
                _ptr_add(self._surf._pixels, data.pixeloffset, c_ubyte)

    def _unprep(self):
        data = self._subsurface
        if data:
            data.owner.unlock()

    def __repr__(self):
        if self._surf:
            if self._surf.flags & SDL_HWSURFACE:
                t = 'HW'
            else:
                t = 'SW'
            return '<Surface(%dx%dx%d %s)>' % \
                (self._surf.w, self._surf.h, self._surf.format.BitsPerPixel, t)
        else:
            return '<Surface(Dead Display)>'

    def __copy__(self):
        return self.copy()

    def blit(self, source, destpos, sourcerect=None, special_flags=0):
        '''Copy a source surface onto this surface.

        The blitting will copy pixels from the source. It will
        respect any special modes like colorkeying and alpha. If hardware
        support is available, it will be used.

        :Parameters:
            `source` : `Surface`
                The source surface to copy from
            `destpost` : (int, int) or `Rect`
                Position to draw the source on the destination surface.  This
                may be either a 2-item sequence or a Rect (in which case the
                size is ignored).
            `sourcerect` : `Rect`
                If specified, the portion of the source surface to copy.  If
                not given, the entire surface will be copied.
            `special_flags` : int
                Optional blend operation; one of BLEND_ADD, BLEND_SUB,
                BLEND_MULT, BLEND_MIN, BLEND_MAX.

        :rtype: Rect
        :return: the actual area blitted.

        :note: `special_flags` is not yet implemented.
        '''

        if self._surf.flags & SDL_OPENGL and \
           not self._surf.flags & SDL_OPENGLBLIT:
            raise pygame.base.error, \
                 'Cannot blit to OPENGL Surfaces (OPENGLBLIT is ok)'
        if sourcerect:
            sourcerect = pygame.rect._rect_from_object(sourcerect)._r

        destrect = SDL_Rect(0, 0, source._surf.w, source._surf.h)
        try:
            destpos = pygame.rect._rect_from_object(destpos)[:2]
        except:
            pass
        destrect.x = destpos[0]
        destrect.y = destpos[1]
        _surface_blit(self, source, destrect, sourcerect, special_flags)
        return pygame.rect.Rect(destrect)

    def convert(self, arg1=None, flags=None):
        '''Create a copy of a surface with different format

        Creates a new copy of the Surface with the pixel format changed. The
        new pixel format can be determined from another existing Surface.
        Otherwise depth, flags, and masks arguments can be used, similar to
        `__init__`.

        If no arguments are passed the new Surface will have the same pixel
        format as the display Surface. This is always the fastest format for
        blitting. It is a good idea to convert all Surfaces before they are
        blitted many times.

        The converted Surface will have no pixel alphas. They will be stripped
        if the original had them. See `convert_alpha` for preserving
        or creating per-pixel alphas

        :Parameters:
            `arg1` : `Surface` or (int, int, int, int), int or None
                If a surface is specified, it will be used to determine the
                target pixel format, and `flags` will be ignored.  If a tuple
                is specified, it is the set of RGBA masks for the new surface.
                Otherwise, this argument is the depth, specified in bits per
                pixel.
            `flags` : int  or None
                  Otherwise, this argument is the flags
                for the new surface; a combination of SWSURFACE, HWSURFACE
                or ASYNCBLIT.

        :rtype: `Surface`
        '''
        if not SDL_WasInit(SDL_INIT_VIDEO):
            raise pygame.base.error, \
                  'cannot convert without pygame.display initialized.'
        
        surf = self._surf
        if surf.flags & SDL_OPENGL:
            raise pygame.base.error, 'Cannot convert opengl display'

        self._prep()

        newsurf = None
        if not arg1:
            newsurf = SDL_DisplayFormat(surf)
        elif isinstance(arg1, Surface):
            src = arg1._surf
            flags = src.flags | \
                    (surf.flags & (SDL_SRCCOLORKEY | SDL_SRCALPHA))
            newsurf = SDL_ConvertSurface(surf, src.format, flags)
        elif type(arg1) in (int, long):
            format = copy(surf.format)
            depth = arg1
            if flags & SDL_SRCALPHA:
                if depth == 16:
                    masks = (0xf << 8, 0xf << 4, 0xf, 0xf << 12)
                elif depth == 32:
                    masks = (0xff << 16, 0xff << 8, 0xff, 0xff << 24)
                else:
                    raise ValueError, \
                    'no standard masks exist for given bitdepth with alpha'
            else:
                if depth == 8:
                    masks = (0xff >> 6 << 5, 0xff >> 5 << 2, 0xff >> 6, 0)
                elif depth == 12:
                    masks = (0xff >> 4 << 8, 0xff >> 4 << 4, 0xff >> 4, 0)
                elif depth == 15:
                    masks = (0xff >> 3 << 10, 0xff >> 3 << 5, 0xff >> 3, 0)
                elif depth == 16:
                    masks = (0xff >> 3 << 11, 0xff >> 2 << 5, 0xff >> 3, 0)
                elif depth == 24 or depth == 32:
                    masks = (0xff << 16, 0xff << 8, 0xff, 0)
                else:
                    raise ValueError, 'nonstandard bit depth given'
        elif type(arg1) in (list, tuple) and len(arg1) == 4:
            masks = arg1
            mask = masks[0] | masks[1] | masks[2] | masks[3]
            depth = 0
            while depth < 32:
                if not mask >> depth:
                    break
                depth += 1

        if not newsurf:
            format.Rmask = masks[0]
            format.Gmask = masks[1]
            format.Bmask = masks[2]
            format.Amask = masks[3]
            format.BitsPerPixel = depth
            format.BytesPerPixel = (depth + 7) / 8
            if not flags:
                flags = surf.flags
            if format.Amask:
                flags |= SDL_SRCALPHA
            newsurf = SDL_ConvertSurface(surf, format, flags)

        self._unprep()
        return Surface(newsurf)

    def convert_alpha(self, surface=None):
        '''Create a copy of a surface with the desired pixel format,
        preserving per-pixel alpha.

        Creates a new copy of the surface with the desired pixel format. The
        new surface will be in a format suited for quick blitting to the given
        format with per pixel alpha. If no surface is given, the new surface
        will be optimized for blittint to the current display.
         
        Unlike the `convert` method, the pixel format for the new
        image will not be exactly the same as the requested source, but it
        will be optimized for fast alpha blitting to the destination.

        :Parameters:
            `surface` : `Surface`
                If specified, the destination surface the pixel format
                will be optimised for.

        :rtype: `Surface`

        :note: The `surface` parameter is currently ignored; the display
            format is always used.
        '''
        if not SDL_WasInit(SDL_INIT_VIDEO):
            raise pygame.base.error, \
                  'cannot convert without pygame.display initialized.'

        self._prep()
        newsurf = SDL_DisplayFormatAlpha(self._surf)
        self._unprep()

        return Surface(newsurf)
        

    def copy(self):
        '''Create a new copy of the surface.

        Makes a duplicate copy of a Surface. The new Surface will have the
        same pixel formats, color palettes, and transparency settings as the
        original.

        :rtype: `Surface`
        '''
        surf = self._surf

        self._prep()
        newsurf = SDL_ConvertSurface(surf, surf.format, surf.flags) 
        self._unprep()

        return Surface(newsurf)

    def fill(self, color, rect=None):
        '''Fill surface with a solid color.

        Fill the Surface with a solid color. If no rect argument is given the
        entire Surface will be filled. The rect argument will limit the fill
        to a specific area. The fill will also be contained by the Surface
        clip area.

        The color argument can be either an RGB sequence or a mapped color
        index.

        :Parameters:
            `color` : (int, int, int) or (int, int, int, int) or int
                Tuple of RGB(A) or mapped color to fill with.
            `rect` : `Rect`
                Area to fill.
                
        :rtype: `Rect`
        :return: the affected surface area.
        '''
        surf = self._surf

        if surf.flags & SDL_OPENGL:
            raise pygame.base.error, 'Cannot call on OPENGL surfaces'

        rgba = pygame.base._rgba_from_obj(color)
        if rgba:
            color = SDL_MapRGBA(surf.format, rgba[0], rgba[1], rgba[2], rgba[3])
        if type(color) not in (int, long):
            raise 'invalid color argument'

        if rect:
            rect = copy(pygame.rect._rect_from_object(rect))
        else:
            rect = pygame.rect.Rect(0, 0, surf.w, surf.h)

        self._prep()
        SDL_FillRect(surf, rect._r, color)
        self._unprep()

        return rect
        
    def set_colorkey(self, color=None, flags=0):
        '''Set the transparent colorkey.

        Set the current color key for the Surface. When blitting this Surface
        onto a destination, and pixels that have the same color as the
        colorkey will be transparent. The color can be an RGB color or a
        mapped color integer. If None is passed, the colorkey will be unset.

        The colorkey will be ignored if the Surface is formatted to use per
        pixel alpha values. The colorkey can be mixed with the full Surface
        alpha value.

        The optional flags argument can be set to pygame.RLEACCEL to provide
        better performance on non accelerated displays. An RLEACCEL Surface
        will be slower to modify, but quicker to blit as a source.

        :Parameters:
            `color` : (int, int, int) or (int, int, int, int) or int or None
                Tuple of RGB(A) or mapped color.  If None, the colorkey is
                unset.
            `flags` : int
                RLEACCEL or 0.

        '''
        surf = self._surf

        if surf.flags & SDL_OPENGL:
            raise pygame.base.error, 'Cannot call on OPENGL surfaces'

        rgba = pygame.base._rgba_from_obj(color)
        if rgba:
            color = SDL_MapRGBA(surf.format, rgba[0], rgba[1], rgba[2], rgba[3])
        if color and type(color) not in (int, long):
            raise 'invalid color argument'
        if color:
            flags |= SDL_SRCCOLORKEY

        SDL_SetColorKey(surf, flags, color)
        
    def get_colorkey(self):
        '''Get the current transparent colorkey.

        :rtype: (int, int, int, int) or None
        :return: The current RGBA colorkey value for the surface, or None
            if the colorkey is not set.
        '''
        surf = self._surf

        if surf.flags & SDL_OPENGL:
            raise pygame.base.error, 'Cannot call on OPENGL surfaces'

        if not surf.flags & SDL_SRCCOLORKEY:
            return None

        return SDL_GetRGBA(surf.format.colorkey, surf.format)

    def set_alpha(self, value=None, flags=0):
        '''Set the alpha value for the full surface.

        Set the current alpha value fo r the Surface. When blitting this
        Surface onto a destination, the pixels will be drawn slightly
        transparent. The alpha value is an integer from 0 to 255, 0 is fully
        transparent and 255 is fully opaque. If None is passed for the alpha
        value, then the Surface alpha will be disabled.

        This value is different than the per pixel Surface alpha. If the
        Surface format contains per pixel alphas, then this alpha value will
        be ignored.  If the Surface contains per pixel alphas, setting the
        alpha value to None will disable the per pixel transparency.

        The optional flags argument can be set to pygame.RLEACCEL to provide
        better performance on non accelerated displays. An RLEACCEL Surface
        will be slower to modify, but quicker to blit as a source.

        :Parameters:
            `value` : int or None
                The alpha value, in range [0, 255].  If None, surface alpha
                is disabled.
            `flags` : int
                RLEACCEL or 0

        '''
        surf = self._surf

        if surf.flags & SDL_OPENGL:
            raise pygame.base.error, 'Cannot call on OPENGL surfaces'

        if value:
            flags |= SDL_SRCALPHA
            value = max(min(255, value), 0)
        else:
            value = 255

        SDL_SetAlpha(surf, flags, alpha)

    def get_alpha(self):
        '''Get the current surface alpha value.

        :rtype: int or None
        :return: The current alpha value for the surface, or None if it
            is not set.
        '''
        surf = self._surf

        if surf.flags & SDL_OPENGL:
            raise pygame.base.error, 'Cannot call on OPENGL surfaces'
        
        if surf.flags & SDL_SRCALPHA:
            return surf.format.alpha

        return None

    def lock(self):
        '''Lock the surface memory for pixel access.

        Lock the pixel data of a Surface for access. On accelerated Surfaces,
        the pixel data may be stored in volatile video memory or nonlinear
        compressed forms. When a Surface is locked the pixel memory becomes
        available to access by regular software. Code that reads or writes
        pixel values will need the Surface to be locked.

        Surfaces should not remain locked for more than necessary. A locked
        Surface can often not be displayed or managed by Pygame.

        Not all Surfaces require locking. The `mustlock` method can
        determine if it is actually required. There is little performance
        penalty for locking and unlocking a Surface that does not need it.

        All pygame functions will automatically lock and unlock the Surface
        data as needed. If a section of code is going to make calls that will
        repeatedly lock and unlock the Surface many times, it can be helpful
        to wrap the block inside a lock and unlock pair.

        It is safe to nest locking and unlocking calls. The surface will only
        be unlocked after the final lock is released.
        '''
        if self._subsurface:
            self._prep()
        SDL_LockSurface(self._surf)

    def unlock(self):
        '''Unlock the surface memory from pixel access.

        Unlock the Surface pixel data after it has been locked. The unlocked
        Surface can once again be drawn and managed by Pygame. See the
        Surface.lock() documentation for more details.

        All pygame functions will automatically lock and unlock the Surface
        data as needed. If a section of code is going to make calls that will
        repeatedly lock and unlock the Surface many times, it can be helpful
        to wrap the block inside a lock and unlock pair.

        It is safe to nest locking and unlocking calls. The surface will only
        be unlocked after the final lock is released.
        '''
        SDL_UnlockSurface(self._surf)
        if self._subsurface:
            self._unprep()

    def mustlock(self):
        '''Test if the surface requires locking.

        Returns True if the Surface is required to be locked to access pixel
        data.  Usually pure software Surfaces do not require locking. This
        method is rarely needed, since it is safe and quickest to just lock
        all Surfaces as needed.

        All pygame functions will automatically lock and unlock the Surface
        data as needed. If a section of code is going to make calls that will
        repeatedly lock and unlock the Surface many times, it can be helpful
        to wrap the block inside a lock and unlock pair.

        :rtype: bool
        '''
        return SDL_MUSTLOCK(self._surf) or self._subsurface != None

    def get_locked(self):
        '''Test if the surface is currently locked.

        Returns True when the Surface is locked. It doesn't matter how many
        times the Surface is locked.
        
        :rtype: bool
        '''
        return self._surf._pixels.contents != None

    def get_at(self, pos):
        '''Get the color value at a single pixel.

        Return the RGBA color value at the given pixel. If the Surface has no
        per pixel alpha, then the alpha value will always be 255 (opaque). If
        the pixel position is outside the area of the Surface an IndexError
        exception will be raised.

        Getting and setting pixels one at a time is generally too slow to be
        used in a game or realtime situation.

        This function will temporarily lock and unlock the Surface as needed.

        :Parameters:
            `pos` : (int, int)
                X, Y coordinates of the pixel to read
        
        :rtype: (int, int, int, int)
        :return: RGBA color
        '''
        surf = self._surf
        if surf.flags & SDL_OPENGL:
            raise pygame.base.error, 'Cannot call on OPENGL surfaces'

        x, y = pos
        if x < 0 or x >= surf.w or y < 0 or y >= surf.h:
            raise pygame.base.error, 'pixel index out of range'

        format = surf.format
        pitch = surf.pitch / format.BytesPerPixel

        self.lock()
        color = surf.pixels[y * pitch + x]
        self.unlock()

        return SDL_GetRGBA(color, format)
        
    def set_at(self, pos, color):
        '''Set the color value for a single pixel.

        Set the RGBA or mapped integer color value for a single pixel. If the
        Surface does not have per pixel alphas, the alpha value is ignored.
        Settting pixels outside the Surface area or outside the Surface
        clipping will have no effect.

        Getting and setting pixels one at a time is generally too slow to be
        used in a game or realtime situation.

        This function will temporarily lock and unlock the Surface as needed.

        :Parameters:
            `pos` : (int, int)
                X, Y coordinates of the pixel to write
            `color` : (int, int, int) or (int, int, int, int) or int
                Tuple of RGB(A) or mapped color
        
        '''
        surf = self._surf
        if surf.flags & SDL_OPENGL:
            raise pygame.base.error, 'Cannot call on OPENGL surfaces'

        x, y = pos
        if x < surf.clip_rect.x or x >= surf.clip_rect.x + surf.clip_rect.w or \
           y < surf.clip_rect.y or y >= surf.clip_rect.y + surf.clip_rect.h:
            return

        format = surf.format
        rgba = pygame.base._rgba_from_obj(color)
        if rgba:
            color = SDL_MapRGBA(format, rgba[0], rgba[1], rgba[2], rgba[3])
        if type(color) not in (int, long):
            raise 'invalid color argument'

        pitch = surf.pitch / format.BytesPerPixel

        self.lock()
        self._surf.pixels[y * pitch + x] = color
        self.unlock()

def _surface_blit(destobj, srcobj, destrect, srcrect, special_flags):
    dst = destobj._surf
    src = srcobj._surf
    subsurface = None
    suboffsetx = 0
    suboffsety = 0
    didconvert = False

    if special_flags:
        raise NotImplementedError, 'TODO'
    
    if destobj._subsurface:
        raise NotImplementedError, 'TODO'

    # Can't blit alpha to 8 bit, creashes SDL
    if dst.format.BytesPerPixel == 1 and \
       (src.format.Amask or src.flags & SDL_SRCALPHA):
        didconvert = True
        src = SDL_DisplayFormat(src)

    if dst.format.Amask and \
       dst.flags & SDL_SRCALPHA and \
       not (src.format.Amask and not (src.flags & SDL_SRCALPHA)) and \
       (dst.format.BytesPerPixel == 2 or dst.format.BytesPerPixel == 4):
        raise NotImplementedError, 'TODO'
    else:
        result = SDL_BlitSurface(src, srcrect, dst, dstrect)

    if didconvert:
        SDL_FreeSurface(src)

    if result == -2:
        raise pygame.base.error, 'Surface was lost'

def _ptr_add(ptr, offset, offset_type):
    return pointer(type(ptr.contents).from_address(\
               addressof(ptr.contents) + sizeof(offset_type) * offset))
