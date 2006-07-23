#!/usr/bin/env python

'''Surface module containing the Surface class.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from copy import copy
import weakref

from SDL import *
import pygame.base
import pygame.locals
import pygame.rect

class _SubSurface_Data(object):
    __slots__ = ['owner', 'pixeloffset', 'offsetx', 'offsety']

class Surface(object):
    __slots__ = ['_surf', '_subsurface', '_weakrefs']

    def __init__(self, size=(0,0), flags=0, depth=0, masks=None, 
                 surf=None, subsurf=None):
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
            `subsurf` : `_SubSurface_Data`
                Used internally.

        '''
        self._weakrefs = []
        if surf:
            if not isinstance(surf, SDL_Surface):
                raise TypeError, 'surf'
            self._surf = surf
            self._subsurface = subsurf
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
                _ptr_add(data.owner._surf._pixels, data.pixeloffset, c_ubyte)

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
        destrect.x = int(destpos[0])
        destrect.y = int(destpos[1])
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
        return Surface(surf=newsurf)

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

        return Surface(surf=newsurf)
        

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

        return Surface(surf=newsurf)

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
        if color is not None:
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

        SDL_SetAlpha(surf, flags, value)

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

    def lifelock(self, obj):
        '''Lock the surface for as long as obj is alive.

        This uses a weak reference to obj to detect when it is garbage
        collected, at which point the surface is unlocked again.
        '''
        self.lock()
        self._weakrefs.append(weakref.ref(obj, self._lifelock_callback))

    def _lifelock_callback(self, ref):
        self._weakrefs.remove(ref)
        self.unlock()

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
        clip_rect = SDL_GetClipRect(surf)
        if x < clip_rect.x or x >= clip_rect.x + clip_rect.w or \
           y < clip_rect.y or y >= clip_rect.y + clip_rect.h:
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

    def get_palette(self):
        '''Get the color index palette for an 8 bit Surface.

        Return a list of up to 256 color elements that represent the indexed
        colors used in an 8bit Surface. The returned list is a copy of the
        palette, and changes will have no effect on the Surface.

        :rtype: list of (int, int, int)
        '''
        pal = self._surf.format.palette
        if not pal:
            raise pygame.base.error, 'Surface has no palette to get'

        return [(c.r, c.g, c.b) for c in pal.colors]

    def get_palette_at(self, index):
        '''Get the color for a single entry in a palette

        Returns the red, green, and blue color values for a single index in a
        Surface palette. The index should be a value from 0 to 255.

        :rtype: (int, int, int)
        '''
        pal = self._surf.format.palette
        if not pal:
            raise pygame.base.error, 'Surface has no palette to get'
        if index < 0 or index >= pal.ncolors:
            raise pygame.base.error, 'index out of bounds'

        c = pal.colors[index]
        return (c.r, c.g, c.b)

    def set_palette(self, palette):
        '''Set the color palette for an 8 bit Surface.

        Set the full palette for an 8bit Surface. This will replace the colors
        in the existing palette. A partial palette can be passed and only the
        first colors in the original palette will be changed.

        This function has no effect on a Surface with more than 8 bits per
        pixel.

        :Parameters:
            `palette` : list of (int, int, int)
                RGB palette values
                
        ''' 
        if not SDL_WasInit(SDL_INIT_VIDEO):
            raise pygame.base.error, \
                  'cannot set palette without pygame.display initialized'

        pal = self._surf.format.palette
        l = min(pal.ncolors, len(palette))
        colors = [SDL_Color(c[0], c[1], c[2]) for c in palette[:l]]
        SDL_SetColors(self._surf, colors, 0)

    def set_palette_at(self, index, color):
        '''Set the color for a single index in an 8 bit Surface palette.

        Set the palette value for a single entry in a Surface palette. The
        index should be a value from 0 to 255.

        This function has no effect on a Surface with more than 8 bits per
        pixel.

        :Parameters:
            `index` : int
                Palette index to set, in range [0, 255]
            `color` : (int, int, int)
                RGB value to set.
        
        '''
        if not SDL_WasInit(SDL_INIT_VIDEO):
            raise pygame.base.error, \
                  'cannot set palette without pygame.display initialized'

        pal = self._surf.format.palette
        if not pal:
            raise pygame.base.error, 'Surface is not paletteized'
        if index < 0 or index >= pal.ncolors:
            raise IndexError, 'index out of bounds'
        
        colors = [SDL_Color(color[0], color[1], color[2])]
        SDL_SetColors(self._surf, colors, index)


    def map_rgb(self, color):
        '''Convert a color into a mapped color value.

        Convert an RGBA color into the mapped integer value for this Surface.
        The returned integer will contain no more bits than the bit depth of
        the Surface.  Mapped color values are not often used inside Pygame,
        but can be passed to most functions that require a Surface and a
        color.

        See the Surface object documentation for more information about colors
        and pixel formats.

        :Parameters:
            `color` : (int, int, int, int)
                RGBA color to map

        :rtype: int
        '''
        surf = self._surf
        rgba = pygame.base._rgba_from_obj(color)
        if not rgba:
            raise pygame.base.error, 'Invalid RGBA argument'
        return SDL_MapRGBA(surf.format, rgba[0], rgba[1], rgba[2], rgba[3])
        
    def unmap_rgb(self, mapped):
        '''Convert a mapped integer color value into a Color.

        Convert an mapped integer color into the RGB color components for this
        Surface.  Mapped color values are not often used inside Pygame, but
        can be passed to most functions that require a Surface and a color.

        See the Surface object documentation for more information about colors
        and pixel formats.

        :Parameters:
            `mapped` : int
                Mapped color to convert

        :rtype: (int, int, int, int)
        :return: RGBA color value
        '''
        surf = self._surf
        return SDL_GetRGBA(mapped, surf.format)
        
    def set_clip(self, rect=None):
        '''Set the current clipping area of the Surface.

        Each Surface has an active clipping area. This is a rectangle that
        represents the only pixels on the Surface that can be modified. If
        None is passed for the rectangle the full Surface will be available
        for changes.

        The clipping area is always restricted to the area of the Surface
        itself. If the clip rectangle is too large it will be shrunk to fit
        inside the Surface.
        
        :Parameters:
            `rect` : Rect
                Clipping area to set.

        '''
        if rect:
            rect = pygame.rect._rect_from_object(sourcerect)._r
        SDL_SetClipRect(self._surf, rect)


    def get_clip(self):
        '''Get the current clipping are of the Surface.

        Return a rectangle of the current clipping area. The Surface will
        always return a valid rectangle that will never be outside the bounds
        of the image.  If the Surface has had None set for the clipping area,
        the Surface will return a rectangle with the full area of the Surface.
        
        :rtype: Rect
        '''
        return pygame.rect.Rect(SDL_GetClipRect(self._surf))


    def subsurface(self, rect):
        '''Create a new surface that references its parent.

        Returns a new Surface that shares its pixels with its new parent. The
        new Surface is considered a child of the original. Modifications to
        either Surface pixels will effect each other. Surface information like
        clipping area and color keys are unique to each Surface.

        The new Surface will inherit the palette, color key, and alpha
        settings from its parent.

        It is possible to have any number of subsurfaces and subsubsurfaces on
        the parent. It is also possible to subsurface the display Surface if
        the display mode is not hardware accelerated.

        See the `get_offset`, `get_parent` to learn more about the state of a
        subsurface.

        :Parameters:
            `rect` : Rect
                Area of the parent surface to use.

        :rtype: `Surface`
        '''
        surf = self._surf
        format = surf.format
        if surf.flags & SDL_OPENGL:
            raise pygame.base.error, 'Cannot call on OPENGL Surfaces'

        rect = pygame.rect._rect_from_object(rect)._r
        if rect.x < 0 or rect.y < 0 or \
           rect.x + rect.w > surf.w or rect.y + rect.y > surf.h:
            raise ValueError, 'subsurface rectangle outside surface area'

        self.lock()
        pixeloffset = rect.x * format.BytesPerPixel + rect.y * surf.pitch
        startpixel = _ptr_add(surf._pixels, pixeloffset, c_ubyte)
        sub = SDL_CreateRGBSurfaceFrom(startpixel, rect.w, rect.h, 
            format.BitsPerPixel, surf.pitch, 
            format.Rmask, format.Gmask, format.Bmask, format.Amask)
        self.unlock()

        if format.BytesPerPixel == 1 and format._palette.contents:
            SDL_SetPalette(sub, SDL_LOGPAL, format.palette.colors, 0)
        if surf.flags & SDL_SRCALPHA:
            SDL_SetAlpha(sub, surf.flags & SDL_SRCALPHA, format.alpha)
        if surf.flags & SDL_SRCCOLORKEY:
            SDL_SetColorKey(sub, 
                surf.flags & (SDL_SRCCOLORKEY | SDL_RLEACCEL), format.colorkey)

        data = _SubSurface_Data()
        data.owner = self
        data.pixeloffset = pixeloffset
        data.offsetx = rect.x
        data.offsety = rect.y
        subobj = Surface(surf=sub, subsurf=data)

        return subobj


    def get_parent(self):
        '''Find the parent of a subsurface.

        Returns the parent Surface of a subsurface. If this is not a
        subsurface then None will be returned.

        :rtype: `Surface` or None
        '''
        subdata = self._subsurface
        if not subdata:
            return None
        return subdata.owner

    def get_abs_parent(self):
        '''Find the top level parent of a subsurface.

        Returns the parent Surface of a subsurface. If this is not a
        subsurface then None will be returned.

        :rtype: `Surface` or None
        '''
        obj = self
        while obj._subsurface:
            obj = obj._subsurface.owner
        return obj

    def get_offset(self):
        '''Find the position of a child subsurface inside a parent.

        Get the offset position of a child subsurface inside of a parent. If
        the Surface is not a subsurface this will return (0, 0).

        :rtype: (int, int)
        '''
        if not self._subsurface:
            return (0, 0)
        return self._subsurface.offsetx, self._subsurface.offsety

    def get_abs_offset(self):
        '''Find the absolute position of a child subsurface inside its top
        level parent.

        Get the offset position of a child subsurface inside of its top level
        parent Surface. If the Surface is not a subsurface this will return
        (0, 0).

        :rtype: (int, int)
        '''
        obj = self
        offsetx = 0
        offsety = 0
        while obj._subsurface:
            offsetx += obj._subsurface.offsetx
            offsety += obj._subsurface.offsety
            obj = obj._subsurface.owner
        return offsetx, offsety


    def get_size(self):
        '''Get the dimensions of the Surface.

        Return the width and height of the Surface in pixels.

        :rtype: (int, int)
        :return: width, height
        '''
        return self._surf.w, self._surf.h

    def get_width(self):
        '''Get the width of the Surface.

        Return the width of the Surface in pixels.
        
        :rtype: int
        '''
        return self._surf.w

    def get_height(self):
        '''Get the height of the Surface.

        Return the height of the Surface in pixels.

        :rtype: int
        '''
        return self._surf.h

    def get_rect(self, **kwargs):
        '''Get the rectangular area of the Surface.

        Returns a new rectangle covering the entire surface. This rectangle
        will always start at 0, 0 with a width. and height the same size as
        the image.
         
        You can pass keyword argument values to this function. These named
        values will be applied to the attributes of the Rect before it is
        returned. An example would be 'mysurf.get_rect(center=(100,100))' to
        create a rectangle for the Surface centered at a given position.
        
        :rtype: Rect
        '''
        rect = pygame.rect.Rect(0, 0, self._surf.w, self._surf.h)
        for key, value in kwargs.items():
            setattr(rect, key, value)
        return rect

    def get_bitsize(self):
        '''Get the bit depth of the Surface pixel format.

        Returns the number of bits used to represent each pixel. This value
        may not exactly fill the number of bytes used per pixel. For example a
        15 bit Surface still requires a full 2 bytes.
        
        :rtype: int
        '''
        return self._surf.format.BitsPerPixel

    def get_bytesize(self):
        '''Get the bytes used per Surface pixel.

        Return the number of bytes used per pixel.
        
        :rtype: int
        '''
        return self._surf.format.BytesPerPixel

    def get_flags(self):
        '''Get the additional flags used for the Surface.

        Returns a set of current Surface features. Each feature is a bit in
        the flags bitmask.  Possible flags are:

        SWSURFACE
            Surface is in system memory
        HWSURFACE
            Surface is in video memory
        ASYNCBLIT
            Use asynchronous blits if possible
        HWACCEL  
            Blit uses hardware acceleration
        SRCCOLORKEY
            Blit uses a source color key
        RLEACCELOK
            Private flag
        RLEACCEL
            Surface is RLE encoded
        SRCALPHA
            Blit uses source alpha blending
        PREALLOC
            Surface uses preallocated memory
        
        :rtype: int
        '''
        return self._surf.flags

    def get_pitch(self):
        '''Get the number of bytes used per Surface row.

        Return the number of bytes separating each row in the Surface.
        Surfaces in video memory are not always linearly packed. Subsurfaces
        will also have a larger pitch than their real width.

        This value is not needed for normal Pygame usage.

        :rtype: int
        '''
        return self._surf.pitch

    def get_masks(self):
        '''Get the bitmasks needed to convert between a color and a mapped
        integer.

        Returns the bitmasks used to isolate each color in a mapped integer.

        This value is not needed for normal Pygame usage.

        :rtype: (int, int, int, int)
        '''
        format = self._surf.format
        return format.Rmask, format.Gmask, format.Bmask, format.Amask
        
    def get_shifts(self):
        '''Get the bit shifts needed to convert between a color and a mapped
        integer. 

        Returns the pixel shifts need to convert between each color and a mapped
        integer.

        This value is not needed for normal Pygame usage.

        :rtype: (int, int, int, int)
        '''
        format = self._surf.format
        return format.Rshift, format.Gshift, format.Bshift, format.Ashift

    def get_losses(self):
        '''Get the significant bits used to convert between a color and a
        mapped integer.

        Return the least significan number of bits stripped from each color
        in a mapped integer.

        This value is not needed for normal Pygame usage.

        :rtype: (int, int, int, int)
        '''
        format = self._surf.format
        return format.Rloss, format.Gloss, format.Bloss, format.Aloss


def _surface_blit(destobj, srcobj, dstrect, srcrect, special_flags):
    dst = destobj._surf
    src = srcobj._surf
    subsurface = None
    suboffsetx = 0
    suboffsety = 0
    didconvert = False

    if destobj._subsurface:
        subdata = destobj._subsurface
        owner = subdata.owner
        subsurface = owner._surf
        suboffsetx = subdata.offsetx
        suboffsety = subdata.offsety

        while owner._subsurface:
            subdata = owner._subsurface
            owner = subdata.owner
            subsurface = owner._surf
            suboffsetx += subdata.offsetx
            suboffsety += subdata.offsety

        orig_clip = SDL_GetClipRect(subsurface)
        sub_clip = SDL_GetClipRect(dst)
        sub_clip.x += suboffsetx
        sub_clip.y += suboffset.y
        SDL_SetClipRect(subsurface, sub_clip)
        dstrect.x += suboffsetx
        dstrect.y += suboffsety
        dst = subsurface
    else:
        destobj._prep()

    srcobj._prep()

    # Can't blit alpha to 8 bit, creashes SDL
    if dst.format.BytesPerPixel == 1 and \
       (src.format.Amask or src.flags & SDL_SRCALPHA):
        didconvert = True
        src = SDL_DisplayFormat(src)

    if dst.format.Amask and \
       dst.flags & SDL_SRCALPHA and \
       not (src.format.Amask and not (src.flags & SDL_SRCALPHA)) and \
       (dst.format.BytesPerPixel == 2 or dst.format.BytesPerPixel == 4):
        result = _software_blit(src, srcrect, dst, dstrect, special_flags)
    elif special_flags:
        result = _software_blit(src, srcrect, dst, dstrect, special_flags)
    else:
        result = SDL_BlitSurface(src, srcrect, dst, dstrect)

    if didconvert:
        SDL_FreeSurface(src)

    if subsurface:
        SDL_SetClipRect(subsurface, orig_clip)
        dstrect.x -= suboffsetx
        dstrect.y -= suboffsety
    else:
        destobj._unprep()
    srcobj._unprep()

    if result == -2:
        raise pygame.base.error, 'Surface was lost'

def _software_blit(src, srcrect, dst, dstrect, special_flags):
    # Clip srcrect against source surface
    if srcrect:
        srcx = srcrect.x
        w = srcrect.w
        if srcx < 0:
            w += srcx
            dstrect.x -= srcx
            srcx = 0
        w = max(src.w - srcx, w)

        srcy = srcrect.y
        h = srcrect.h
        if srcy < 0:
            h += srcy
            dstrect.y -= srcy
            srcy = 0
        h = max(src.h - srcy, h)
    else:
        srcx = srcy = 0
        w = src.w
        h = src.h

    # Clip destination rect against clip rectangle
    clip = dst.clip_rect
    dx = clip.x - dstrect.x
    if dx > 0:
        w -= dx
        dstrect.x += dx
        srcx += dx
    dx = dstrect.x + w - clip.x - clip.w
    if dx > 0:
        w -= dx

    dy = clip.y - dstrect.y
    if dy > 0:
        h -= dy
        dstrect.y += dy
        srcy += dy
    dy = dstrect.y + h - clip.y - clip.h
    if dy > 0:
        h -= dy

    # No blit required
    if w <= 0 or h <= 0:
        dstrect.w = dstrect.h = 0
        return

    # Update destination rect (for caller)
    dstrect.w = w
    dstrect.h = h

    # New to Pygame-ctypes
    assert dst.format.BytesPerPixel > 1
    free_src = False
    if src.format.BytesPerPixel == 1:
        # XXX easy way out; could be faster using array module
        src = SDL_ConvertSurface(src, dst.format, 0)
        free_src = True

    # Mmm, fun times...
    srcRmask = src.format.Rmask
    srcGmask = src.format.Gmask
    srcBmask = src.format.Bmask
    srcAmask = src.format.Amask
    srcRshift = src.format.Rshift
    srcGshift = src.format.Gshift
    srcBshift = src.format.Bshift
    srcAshift = src.format.Ashift
    srcRloss = src.format.Rloss
    srcGloss = src.format.Gloss
    srcBloss = src.format.Bloss
    srcAloss = src.format.Aloss
    dstRmask = dst.format.Rmask
    dstGmask = dst.format.Gmask
    dstBmask = dst.format.Bmask
    dstAmask = dst.format.Amask
    dstRshift = dst.format.Rshift
    dstGshift = dst.format.Gshift
    dstBshift = dst.format.Bshift
    dstAshift = dst.format.Ashift
    dstRloss = dst.format.Rloss
    dstGloss = dst.format.Gloss
    dstBloss = dst.format.Bloss
    dstAloss = dst.format.Aloss

    srcpitch = src.pitch / src.format.BytesPerPixel
    dstpitch = dst.pitch / dst.format.BytesPerPixel
    srcpitchdelta = srcpitch - w
    dstpitchdelta = dstpitch - w
    srci = srcy * srcpitch + srcx
    dsti = dstrect.y * dstpitch + dstrect.x

    src24 = src.format.BitsPerPixel == 24
    dst24 = dst.format.BitsPerPixel == 24
    if src24:
        srcpitch = src.pitch
        srci = srcy * srcpitch + srcx * 3
        srcpitchdelta = srcpitch - w * 3
    if dst24:
        dstpitch = dst.pitch
        dsti = dsty * dstpitch + dstx * 3
        dstpitchdelta = dstpitch - w * 3

    # Both surfaces are already prepped by caller, just need to lock
    SDL_LockSurface(src)
    SDL_LockSurface(dst)

    if src.pixels.have_array():
        # 2D plane manipulation with numpy, Numeric or numarray
        if src.pixels.have_numpy():
            src2d = src.pixels.as_numpy((src.h, srcpitch))
            dst2d = dst.pixels.as_numpy((dst.h, dstpitch))
            import numpy
            array = numpy
            copy_array = False
        else:
            src2d = src.pixels.to_array((src.h, srcpitch))
            dst2d = dst.pixels.to_array((dst.h, dstpitch))
            array = src.pixels.array_module()
            copy_array = True

        if src24:
            src2d_rect = src2d[srcy:srcy+h,srcx:srcx+w*3]
            sR = src2d_rect[:,::3]
            sG = src2d_rect[:,1::3]
            sB = src2d_rect[:,2::3]
            sA = 255
        else:
            src2d_rect = src2d[srcy:srcy+h,srcx:srcx+w]
            sR = ((src2d_rect & srcRmask) >> srcRshift) << srcRloss
            sG = ((src2d_rect & srcGmask) >> srcGshift) << srcGloss
            sB = ((src2d_rect & srcBmask) >> srcBshift) << srcBloss
            sA = ((src2d_rect & srcAmask) >> srcAshift) << srcAloss

        if dst24:
            # XXX TODO This is completely untested.  The planes probably
            # need to be cast to a larger data type.
            dst2d_rect = dst2d[dstrect.y:dstrect.y+h,dstrect.x:dstrect.x+w*3]
            dR = dst2d_rect[:,::3]
            dG = dst2d_rect[:,1::3]
            dB = dst2d_rect[:,2::3]
            dA = 255
        else:
            dst2d_rect = dst2d[dstrect.y:dstrect.y+h,dstrect.x:dstrect.x+w]
            dR = ((dst2d_rect & dstRmask) >> dstRshift) << dstRloss
            dG = ((dst2d_rect & dstGmask) >> dstGshift) << dstGloss
            dB = ((dst2d_rect & dstBmask) >> dstBshift) << dstBloss
            dA = ((dst2d_rect & dstAmask) >> dstAshift) << dstAloss

        # Perform blend
        if special_flags == 0:
            if src.flags & SDL_SRCALPHA and src.format.Amask:
                pass    # keep sA
            elif src.flags & SDL_SRCCOLORKEY:
                sA = array.equal(src2d_rect, src.format.colorkey) * \
                     src.format.alpha
            else:
                sA = src.format.alpha

            comparison = array.equal(dA, 0)
            dR = array.choose( comparison,
                         ( ((dR << 8) + (sR - dR) * sA + sR) >> 8, 
                           sR ) )
            dG = array.choose( comparison,
                         ( ((dG << 8) + (sG - dG) * sA + sG) >> 8, 
                           sG ) )
            dB = array.choose( comparison,
                         ( ((dB << 8) + (sB - dB) * sA + sB) >> 8, 
                           sB ) )
            dA = array.choose( comparison,
                         ( sA + dA - sA * dA / 255,
                           sA ) )
        elif special_flags == pygame.locals.BLEND_ADD:
            dR = array.minimum(dR + sR, 255)
            dG = array.minimum(dG + sG, 255)
            dB = array.minimum(dB + sB, 255)
        elif special_flags == pygame.locals.BLEND_SUB:
            dR = array.choose(array.greater(dR, sR), 
                              (0, dR - sR))
            dG = array.choose(array.greater(dG, sG), 
                              (0, dG - sG))
            dB = array.choose(array.greater(dB, sB), 
                              (0, dB - sB))
        elif special_flags == pygame.locals.BLEND_MULT:
            dR = (dR * sR) >> 8
            dG = (dG * sG) >> 8
            dB = (dB * sB) >> 8
        elif special_flags == pygame.locals.BLEND_MIN:
            dR = array.minimum(dR, sR)
            dG = array.minimum(dG, sG)
            dB = array.minimum(dB, sB)
        elif special_flags == pygame.locals.BLEND_MAX:
            dR = array.maximum(dR, sR)
            dG = array.maximum(dG, sG)
            dB = array.maximum(dB, sB)
        else:
            raise ValueError, 'Unknown blend flag %d' % special_flags

        # Finished, set result.
        if dst24:
            dst2d_rect[:,::3] = dR
            dst2d_rect[:,1::3] = dG
            dst2d_rect[:,2::3] = dB
        else:
            dst2d_rect[:,:] = ((dR >> dstRloss) << dstRshift) | \
                              ((dG >> dstGloss) << dstGshift) | \
                              ((dB >> dstBloss) << dstBshift) | \
                              ((dA >> dstAloss) << dstAshift)

        if copy_array:
            dst.pixels.from_array(dst2d)

    else:
        # Slow, simple Python loop (no array module available)
        srcdata = src.pixels.as_ctypes()
        dstdata = dst.pixels.as_ctypes()

        # Choose blend function
        if special_flags == 0:
            if src.flags & SDL_SRCALPHA and src.format.Amask:
                def blend(sr, sg, sb, sa, dr, dg, db, da, srcpixel):
                    if da:
                        return ( ((dr << 8) + (sr - dr) * sa + sr) >> 8,
                                 ((dg << 8) + (sg - dg) * sa + sg) >> 8,
                                 ((db << 8) + (sb - db) * sa + sb) >> 8,
                                 sa + da - sa * da / 255 )
                    else:
                        return sr, sg, sb, sa

            elif src.flags & SDL_SRCCOLORKEY:
                alpha = src.format.alpha
                colorkey = src.format.colorkey
                def blend(sr, sg, sb, sa, dr, dg, db, da, srcpixel):
                    sa = (srcpixel == colorkey) * alpha
                    if da:
                        return ( ((dr << 8) + (sr - dr) * sa + sr) >> 8,
                                 ((dg << 8) + (sg - dg) * sa + sg) >> 8,
                                 ((db << 8) + (sb - db) * sa + sb) >> 8,
                                 sa + da - sa * da / 255 )
                    else:
                        return sr, sg, sb, sa
            else:
                # Use surface alpha
                alpha = src.format.alpha
                def blend(sr, sg, sb, sa, dr, dg, db, da, srcpixel):
                    sa = alpha
                    if da:
                        return ( ((dr << 8) + (sr - dr) * sa + sr) >> 8,
                                 ((dg << 8) + (sg - dg) * sa + sg) >> 8,
                                 ((db << 8) + (sb - db) * sa + sb) >> 8,
                                 sa + da - sa * da / 255 )
                    else:
                        return sr, sg, sb, sa
        elif special_flags == pygame.locals.BLEND_ADD:
            def blend(sr, sg, sb, sa, dr, dg, db, da, srcpixel):
                return ( min(dr + sr, 255),
                         min(dg + sg, 255),
                         min(db + sb, 255),
                         da )
        elif special_flags == pygame.locals.BLEND_SUB:
            def blend(sr, sg, sb, sa, dr, dg, db, da, srcpixel):
                return ( max(dr - sr, 0),
                         max(dg - sg, 0),
                         max(db - sb, 0),
                         da )
        elif special_flags == pygame.locals.BLEND_MULT:
            def blend(sr, sg, sb, sa, dr, dg, db, da, srcpixel):
                return ( (dr * sr) >> 8,
                         (dg * sg) >> 8,
                         (db * sb) >> 8,
                         da )
        elif special_flags == pygame.locals.BLEND_MIN:
            def blend(sr, sg, sb, sa, dr, dg, db, da, srcpixel):
                return ( min(dr, sr),
                         min(dg, sg),
                         min(db, sb),
                         da )
        elif special_flags == pygame.locals.BLEND_MAX:
            def blend(sr, sg, sb, sa, dr, dg, db, da, srcpixel):
                return ( max(dr, sr),
                         max(dg, sg),
                         max(db, sb),
                         da )    
        else:
            raise ValueError, 'Unknown blend flag %d' % special_flags

        y = 0
        while y < h:
            x = 0
            while x < w:
                if src24:
                    srccol = SDL_SwapLE32(srcdata[srci] | \
                                          srcdata[srci+1] << 8 | \
                                          srcdata[srci+2] << 16 | \
                                          0xff << 24)
                    srci += 2
                else:
                    srccol = srcdata[srci]
                if dst24:
                    dstcol = SDL_SwapLE32(dstdata[dsti] | \
                                          dstdata[dsti+1] << 8 | \
                                          dstdata[dsti+2] << 16)
                else:
                    dstcol = dstdata[dsti]
                dR, dG, dB, dA = \
                    blend( ((srccol & srcRmask) >> srcRshift) << srcRloss,
                           ((srccol & srcGmask) >> srcGshift) << srcGloss,
                           ((srccol & srcBmask) >> srcBshift) << srcBloss,
                           ((srccol & srcAmask) >> srcAshift) << srcAloss,
                           ((dstcol & dstRmask) >> dstRshift) << dstRloss,
                           ((dstcol & dstGmask) >> dstGshift) << dstGloss,
                           ((dstcol & dstBmask) >> dstBshift) << dstBloss,
                           ((dstcol & dstAmask) >> dstAshift) << dstAloss,
                           srccol )
                if dst24:
                    # XXX assuming RGB
                    dstdata[dsti] = dR
                    dstdata[dsti + 1] = dG
                    dstdata[dsti + 2] = dB
                    dsti += 2
                else:
                    dstdata[dsti] = ((dR >> dstRloss) << dstRshift) | \
                                    ((dG >> dstGloss) << dstGshift) | \
                                    ((dB >> dstBloss) << dstBshift) | \
                                    ((dA >> dstAloss) << dstAshift)
                srci += 1
                dsti += 1
                x += 1
            y += 1
            srci += srcpitchdelta
            dsti += dstpitchdelta
    
    SDL_UnlockSurface(dst)
    SDL_UnlockSurface(src)

    if free_src:
        SDL_FreeSurface(src)

    return 0

def _ptr_add(ptr, offset, offset_type):
    return pointer(type(ptr.contents).from_address(\
               addressof(ptr.contents) + sizeof(offset_type) * offset))
