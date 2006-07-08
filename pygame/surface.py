#!/usr/bin/env python

'''Surface module containing the Surface class.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from copy import copy

from SDL import *
import pygame.base
import pygame.rect

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

        if not arg1:
            return Surface(SDL_DisplayFormat(surf))
        elif isinstance(arg1, Surface):
            src = arg1._surf
            flags = src.flags | \
                    (surf.flags & (SDL_SRCCOLORKEY | SDL_SRCALPHA))
            return Surface(SDL_ConvertSurface(surf, src.format, flags))
        elif type(arg1) == int:
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
        return Surface(SDL_ConvertSurface(surf, format, flags))

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

        return Surface(SDL_DisplayFormatAlpha(self._surf))
        

    def copy(self):
        '''Create a new copy of the surface.

        Makes a duplicate copy of a Surface. The new Surface will have the
        same pixel formats, color palettes, and transparency settings as the
        original.

        :rtype: `Surface`
        '''
        surf = self._surf
        return Surface(SDL_ConvertSurface(surf, surf.format, surf.flags))

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

        SDL_FillRect(surf, rect._r, color)
        return rect
        



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
