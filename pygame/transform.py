#!/usr/bin/env python

'''Pygame module to transform surfaces.

A Surface transform is an operation that moves or resizes the pixels. All
these functions take a Surface to operate on and return a new Surface with
the results.

Some of the transforms are considered destructive. These means every time
they are performed they lose pixel data. Common examples of this are resizing
and rotating. For this reason, it is better to retransform the original surface than to 
keep transforming an image multiple times. (For example, suppose you are animating
a bouncing spring which expands and contracts. If you applied the size changes
incrementally to the previous images, you would lose detail. Instead, always
begin with the original image and scale to the desired size.)
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import re

from SDL import *

import pygame.base
import pygame.surface

try:
    import Image
    _have_PIL = True
except ImportError:
    _have_PIL = False

def _newsurf_fromsurf(surf, width, height):
    format = surf.format
    newsurf = SDL_CreateRGBSurface(surf.flags, width, height,
        format.BitsPerPixel, 
        format.Rmask, format.Gmask, format.Bmask, format.Amask)
    
    if format.BytesPerPixel == 1 and format._palette:
        SDL_SetColors(newsurf, format.palette.colors, 0)
    if surf.flags & SDL_SRCCOLORKEY:
        SDL_SetColorKey(newsurf, (surf.flags & SDL_RLEACCEL) | SDL_SRCCOLORKEY,
                        format.colorkey)
    if surf.flags & SDL_SRCALPHA:
        SDL_SetAlpha(newsurf, surf.flags, format.alpha)

    return newsurf

def flip(surface, x=False, y=False):
    '''Flip vertically and horizontally.

    This can flip a Surface either vertically, horizontally, or both. Flipping
    a Surface is nondestructive and returns a new Surface with the same
    dimensions.
    
    :Parameters:
        `surface` : `Surface`
            Surface to flip.
        `x` : bool
            If True, the surface will be flipped horizontally.
        `y` : bool
            If True, the surface will be flipped vertically.

    :rtype: `Surface`
    '''
    # Currently implemented by doing regular expressions over string data,
    # which is pretty fast (much faster than working with lists of ints).

    surf = surface._surf
    newsurf = _newsurf_fromsurf(surf, surf.w, surf.h)

    surface.lock()
    data = surf.pixels.to_string()
    surface.unlock()

    rows = re.findall('.' * surf.pitch, data, re.DOTALL)
    if newsurf.pitch < surf.pitch:
        for i in range(len(rows)):
            rows[i] = rows[i][:newsurf.pitch]
    elif newsurf.pitch > surf.pitch:
        pad = '\000' * (newsurf.pitch - surf.pitch)
        for i in range(len(rows)):
            rows[i] = rows[i] + pad

    if y:
        rows.reverse()
    if x:
        pattern = re.compile('.' * surf.format.BytesPerPixel, re.DOTALL)
        for i in range(len(rows)):
            pixels = pattern.findall(rows[i])
            pixels.reverse()
            rows[i] = ''.join(pixels)
    data = ''.join(rows)

    SDL_LockSurface(newsurf)
    memmove(newsurf.pixels.ptr, data, len(data))
    SDL_UnlockSurface(newsurf)

    return pygame.surface.Surface(surf=newsurf)

def scale(surface, size, dest=None):
    '''Resize to new resolution.

    Resizes the Surface to a new resolution. This is a fast scale operation
    that does not sample the results. 

    An optional destination surface can be used, rather than have it create 
    a new one.  This is quicker if you want to repeatedly scale something.  
    However the destination must be the same size as the (width, height) passed 
    in.  Also the destination surface must be the same format.
    
    :Parameters:
        `surface` : `Surface`
            Surface to resize.
        `size` : int, int
            Width, height to resize to.
        `dest` : `Surface`
            Optional destination surface to write to.

    :rtype: `Surface`
    '''
    if not _have_PIL:
        raise NotImplementedError, 'Python imaging library (PIL) required.'
    
    # XXX: Differ from Pygame: subsurfaces permitted.
    width, height = size
    if width < 0 or height < 0:
        raise ValueError, 'Cannot scale to negative size'

    surf = surface._surf
    if not dest:
        newsurf = _newsurf_fromsurf(surf, width, height)
    else:
        dest._prep()
        newsurf = dest._surf

    if newsurf.w != width or newsurf.h != height:
        raise ValueError, 'Destination surface not the given width or height.'

    if newsurf.format.BytesPerPixel != surf.format.BytesPerPixel:
        raise ValueError, \
              'Source and destination surfaces need the same format.'

    if surf.format.BitsPerPixel == 8:
        mode = 'P'
    elif surf.format.BitsPerPixel == 24:
        mode = 'RGB'
    elif surf.format.BitsPerPixel == 32:
        mode = 'RGBA'
    else:
        raise ValueError, 'Unsupported pixel format' # TODO convert

    if width and height:
        surface.lock()
        data = surf.pixels.to_string()
        surface.unlock()

        source_pitch = surf.w * surf.format.BytesPerPixel
        if surf.pitch > source_pitch:
            rows = re.findall('.' * surf.pitch, data, re.DOTALL)
            for i in range(len(rows)):
                rows[i] = rows[i][:source_pitch]
            data = ''.join(rows)

        image = Image.fromstring(mode, (surf.w, surf.h), data)
        image = image.resize((width, height), Image.NEAREST)
        data = image.tostring()

        dest_pitch = width * newsurf.format.BytesPerPixel
        if newsurf.pitch > dest_pitch:
            rows = re.findall('.' * dest_pitch, data, re.DOTALL)
            pad = '\000' * (newsurf.pitch - dest_pitch)
            for i in range(len(rows)):
                rows[i] = rows[i] + pad
            data = ''.join(rows)

        SDL_LockSurface(newsurf)
        memmove(newsurf.pixels.ptr, data, len(data))
        SDL_UnlockSurface(newsurf)

    if dest:
        dest._unprep()
        return dest
    else:
        return pygame.surface.Surface(surf=newsurf)

def rotate(surface, angle):
    '''Rotate an image.

    Unfiltered counterclockwise rotation. The angle argument represents degrees
    and can be any floating point value. Negative angle amounts will rotate
    clockwise.

    Unless rotating by 90 degree increments, the image will be padded larger
    to hold the new size. If the image has pixel alphas, the padded area will
    be transparent. Otherwise pygame will pick a color that matches the Surface
    colorkey or the topleft pixel value.
    
    :Parameters:
        `surface` : `Surface`
            Surface to rotate.
        `angle` : float
            Degrees to rotate anticlockwise.

    :rtype: `Surface`
    '''

def rotozoom(surface, angle, scale):
    '''Filtered scale and rotation.

    This is a combined scale and rotation transform. The resulting Surface will
    be a filtered 32-bit Surface. The scale argument is a floating point value
    that will be multiplied by the current resolution. The angle argument is
    a floating point value that represents the counterclockwise degrees to
    rotate. A negative rotation angle will rotate clockwise.
    
    :Parameters:
        `surface` : `Surface`
            Surface to transform.
        `angle` : float
            Degrees to rotate anticlockwise.
        `scale` : float
            Scale to apply to surface size.
    '''

def scale2x(surface, dest=None):
    '''Specialized image doubler.

    This will return a new image that is double the size of the original. It
    uses the AdvanceMAME Scale2X algorithm which does a 'jaggie-less' scale of
    bitmap graphics.
     
    This really only has an effect on simple images with solid colors. On
    photographic and antialiased images it will look like a regular unfiltered
    scale.

    An optional destination surface can be used, rather than have it create 
    a new one.  This is quicker if you want to repeatedly scale something.  
    However the destination must be twice the size of the source surface passed 
    in.  Also the destination surface must be the same format.
    
    :Parameters:
        `surface` : `Surface`
            Surface to resize.
        `dest` : `Surface`
            Optional destination surface to write to.

    :rtype: `Surface`
    '''
    # XXX differ from Pygame: allow subsurfaces.
    surf = surface._surf
    if not dest:
        newsurf = _newsurf_fromsurf(surf, surf.w * 2, surf.h * 2)
    else:
        dest._prep()
        newsurf = dest._surf

    if newsurf.w != surf.w * 2 or newsurf.h != surf.h * 2:
        raise ValueError, 'Destination surface not 2x bigger.'

    if surf.format.BytesPerPixel != newsurf.format.BytesPerPixel:
        raise ValueError, \
              'Source and destination surfaces need the same format.'

    width, height = surf.w, surf.h
    source_pitch = surf.pitch / surf.format.BytesPerPixel
    dest_pitch = newsurf.pitch / newsurf.format.BytesPerPixel

    surface.lock()
    SDL_LockSurface(newsurf)
    
    srcpix = surf.pixels.as_ctypes()
    dstpix = newsurf.pixels.as_ctypes()
    for y in range(surf.h):
        b_y = max(0, y - 1) * source_pitch
        mid_y = y * source_pitch
        h_y = min(height - 1, y + 1) * source_pitch
        dest_y = y * 2 * dest_pitch
        for x in range(surf.w):
            twox = 2 * x
            b = srcpix[b_y + x]
            d = srcpix[mid_y + max(0, x - 1)]
            e = srcpix[mid_y + x]
            f = srcpix[mid_y + min(width - 1, x + 1)]
            h = srcpix[h_y + x]
            if b != h and d != f:
                if d == b: 
                    dstpix[dest_y + twox] = d
                else: 
                    dstpix[dest_y + twox] = e
                if b == f:
                    dstpix[dest_y + twox + 1] = b
                else:
                    dstpix[dest_y + twox + 1] = e
                if d == h:
                    dstpix[dest_y + dest_pitch + twox] = d
                else:
                    dstpix[dest_y + dest_pitch + twox] = e
                if h == f:
                    dstpix[dest_y + dest_pitch + twox + 1] = h
                else:
                    dstpix[dest_y + dest_pitch + twox + 1] = e
            else:
                dstpix[dest_y + twox] = e
                dstpix[dest_y + twox + 1] = e
                dstpix[dest_y + dest_pitch + twox] = e
                dstpix[dest_y + dest_pitch + twox + 1] = e

    SDL_UnlockSurface(newsurf)
    surface.unlock()

    if dest:
        dest._unprep()
        return dest
    else:
        return pygame.surface.Surface(surf=newsurf)

def chop(surface, rect):
    '''Extract a rectangular area of an image.

    Extracts a portion of an image. All vertical and
    horizontal pixels surrounding the given rectangle area are removed. The
    resulting image is shrunken by the size of pixels removed.
    (The original image is not altered by this operation.)

    :Parameters:
        `surface` : `Surface`
            Surface to crop.
        `rect` : `Rect`
            Area to extract.

    :rtype: `Surface`
    '''
