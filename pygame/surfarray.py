#!/usr/bin/env python

'''Pygame module for accessing surface pixel data.

Functions to convert pixel data between pygame Surfaces and Numeric arrays.
This module will only be available when pygame can use the external Numeric
package.

Note, that surfarray is an optional module.  It requires that Numeric is 
installed to be used.  If not installed, an exception will be raised when
it is used.  eg. NotImplementedError: surfarray module not available

Every pixel is stored as a single integer value to represent the red, green,
and blue colors. The 8bit images use a value that looks into a colormap. Pixels
with higher depth use a bit packing process to place three or four values into
a single number.

The Numeric arrays are indexed by the X axis first, followed by the Y axis.
Arrays that treat the pixels as a single integer are referred to as 2D arrays.
This module can also separate the red, green, and blue color values into
separate indices. These types of arrays are referred to as 3D arrays, and the
last index is 0 for red, 1 for green, and 2 for blue.

Numeric does not use unsigned 16bit integers, images with 16bit data will
be treated as signed integers.

By default Numeric arrays will be returned, however you can use numpy or
numarray instead; see the `set_array_module` function.  Any type of array
(numpy, numarray or Numeric) can be used as the input to any function
regardless of the array module set.  

:note: numarray support is a bit flakey; there are some operations it does not
    support well.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from ctypes import *
import re
import sys

from SDL import *

import pygame.array
import pygame.surface

def array2d(surface):
    '''Copy pixels into a 2d array.

    Copy the pixels from a Surface into a 2D array. The bit depth of the
    surface will control the size of the integer values, and will work for any
    type of pixel format.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock() method).

    :Parameters:
        `surface` : `Surface`
            Surface to copy.

    :rtype: Numeric array
    '''
    pygame.array._check_array()
    _array = pygame.array._array

    surf = surface._surf
    bpp = surf.format.BytesPerPixel

    if bpp <= 0 or bpp > 4:
        raise ValueError, 'unsupport bit depth for surface array'

    surface.lock()
    data = surf.pixels.to_string()
    surface.unlock()

    # Remove extra pitch from each row
    pitchdiff = surf.pitch - surf.w * bpp
    if pitchdiff > 0:
        pattern = re.compile('(%s)%s' % ('.' * surf.w * bpp, '.' * pitchdiff),
                             flags=re.DOTALL)
        data = ''.join(pattern.findall(data))

    if bpp == 3:
        # Pad each triplet of bytes with another zero
        pattern = re.compile('...', flags=re.DOTALL)
        data = '\0'.join(pattern.findall(data))
        if SDL_BYTEORDER == SDL_LIL_ENDIAN:
            data += '\0'
        else:
            data = '\0' + data
        bpp = 4

    shape = surf.h, surf.w
    ar = pygame.array._array_from_string(data, bpp, shape)

    return _array.transpose(ar)

def pixels2d(surface):
    '''Reference pixels into a 2d array.

    Create a new 2D array that directly references the pixel values in a
    Surface.  Any changes to the array will affect the pixels in the Surface.
    This is a fast operation since no data is copied.

    Pixels from a 24-bit Surface cannot be referenced, but all other Surface
    bit depths can.

    The Surface this references will remain locked for the lifetime of the
    array (see the Surface.lock() method).

    :Parameters:
         `surface` : `Surface`
            Surface to reference.

    :rtype: Numeric array
    '''
    surf = surface._surf
    bpp = surf.format.BytesPerPixel

    if bpp == 3 or bpp < 1 or bpp > 4:
        raise ValueError, 'unsupprt bit depth for 2D reference array'

    if surf.pitch % bpp != 0:
        #TODO hack stride
        raise NotImplementedError, "Can't correct for this surface pitch"

    shape = surf.h, surf.pitch / bpp

    surface.lock()
    array = pygame.array._array_from_buffer(surf.pixels.as_ctypes(), bpp, shape)
    surface.lifelock(array)
    surface.unlock()

    array = array[:,:surf.w]
    return pygame.array._array.transpose(array)

def array3d(surface):
    '''Copy pixels into a 3d array.

    Copy the pixels from a Surface into a 3D array.  Arrays are indexed
    by [x,y,colour], where colour is 0 for the red component, 1 for the
    green component and 2 for the blue component.

    The pixel format of the surface will control the maximum size of the
    integer values.

    This function will temporarily lock the Surface as pixels are copied (see
    the Surface.lock() method).

    :note: This function requires Numeric or numpy if a palettized surface
        is used; numarray will fail.

    :Parameters:
         `surface` : `Surface`
            Surface to copy.

    :rtype: Numeric array
    '''
    array = array2d(surface)
    _array = pygame.array._array

    surf = surface._surf
    format = surf.format
    bpp = format.BytesPerPixel

    if format.BytesPerPixel == 1:
        # XXX Fails in numarray:
        pal_r = _array.array([c.r for c in surf.format.palette.colors])
        pal_g = _array.array([c.g for c in surf.format.palette.colors])
        pal_b = _array.array([c.b for c in surf.format.palette.colors])
        # (ValueError: _operator_compute: too many inputs + outputs
        planes = [_array.choose(array, pal_r),
                  _array.choose(array, pal_g),
                  _array.choose(array, pal_b)]
        array = _array.array(planes)
        array = _array.transpose(array, (1, 2, 0))
    else:
        planes = [((array & format.Rmask) >> format.Rshift) << format.Rloss,
                  ((array & format.Gmask) >> format.Gshift) << format.Gloss,
                  ((array & format.Bmask) >> format.Bshift) << format.Bloss] 
        if _array.__name__ == 'Numeric':
            # Workaround bug: if type unspecified, Numeric gives type 'O'
            array = _array.array(planes , _array.UInt8)
        else:
            array = _array.array(planes)
        array = _array.transpose(array, (1, 2, 0))

    return array

def pixels3d(surface):
    '''Reference pixels into a 3d array.

    Create a new 3D array that directly references the pixel values in a
    Surface.  Any changes to the array will affect the pixels in the Surface.
    This is a fast operation since no data is copied.

    This will only work on Surfaces that have 24-bit or 32-bit formats. Lower
    pixel formats cannot be referenced.

    The Surface this references will remain locked for the lifetime of the array
    (see the Surface.lock() method).

    :Parameters:
         `surface` : `Surface`
            Surface to reference.

    :rtype: Numeric array
    '''  
    surf = surface._surf
    bpp = surf.format.BytesPerPixel

    if bpp <= 2 or bpp > 4:
        raise ValueError, 'unsupport bit depth for alpha array'

    if SDL_SwapLE32(surf.format.Rmask) == 0xff << 16 and \
       SDL_SwapLE32(surf.format.Gmask) == 0xff << 8 and \
       SDL_SwapLE32(surf.format.Bmask) == 0xff:
        start = 2
        step = -1
        end = None
    elif SDL_SwapLE32(surf.format.Rmask) == 0xff and \
         SDL_SwapLE32(surf.format.Gmask) == 0xff << 8 and \
         SDL_SwapLE32(surf.format.Bmask) == 0xff << 16:
        start = 0
        step = 1
        end = 3
    else:
        raise ValueError, 'unsupport colormasks for 3D reference array'

    shape = surf.h, surf.pitch 

    surface.lock()
    array = pygame.array._array_from_buffer(surf.pixels.as_bytes().as_ctypes(), 
                                            1, shape)
    surface.lifelock(array)
    surface.unlock()

    array = array[:,:surf.w*bpp]
    array = pygame.array._array.reshape(array, (surf.h, surf.w, bpp))
    array = array[:,:,start:end:step]
    return pygame.array._array.transpose(array, (1, 0, 2))

def array_alpha(surface):
    '''Copy pixel alphas into a 2d array.

    Copy the pixel alpha values (degree of opacity) from a Surface into a
    2D array. This will work for any type of Surface format. Surfaces without
    a pixel alpha will return an array with all opaque values.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock() method).

    :Parameters:
         `surface` : `Surface`
            Surface to copy.

    :rtype: Numeric, numpy or numarray array
    '''
    array = array2d(surface)

    format = surface._surf.format
    if (not format.Amask) or format.BytesPerPixel == 1:
        array[:,:] = 0xff
    else:
        array = array >> format.Ashift << format.Aloss

    return array.astype(pygame.array._array.UInt8)

def pixels_alpha(surface):
    '''Reference pixel alphas into a 2d array.

    Create a new 2D array that directly references the alpha values (degree of
    transparency) in a Surface.  Any changes to the array will affect the
    pixels in the Surface. This is a fast operation since no data is copied.

    This can only work on 32-bit Surfaces with a per-pixel alpha value.

    The Surface this references will remain locked for the lifetime of the
    array.

    :Parameters:
         `surface` : `Surface`
            Surface to reference.

    :rtype: Numeric array
    ''' 
    surf = surface._surf

    if surf.format.BytesPerPixel != 4:
        raise ValueError, 'unsupport bit depth for alpha array'

    if SDL_SwapLE32(surf.format.Amask) == 0xff << 24:
        startpixel = 3
    else:
        startpixel = 0

    shape = surf.h, surf.pitch

    surface.lock()
    array = pygame.array._array_from_buffer(surf.pixels.as_bytes().as_ctypes(), 
                                            1, shape)
    surface.lifelock(array)
    surface.unlock()

    array = array[:,:surf.w*4]
    array = array[:,startpixel::4]
    return pygame.array._array.transpose(array)

def array_colorkey(surface):
    '''Copy the colorkey values into a 2d array.

    Create a new array with the colorkey transparency value from each pixel. If
    the pixel matches the colorkey it will be fully tranparent; otherwise it
    will be fully opaque.

    This will work on any type of Surface format. If the image has no colorkey
    a solid opaque array will be returned.

    This function will temporarily lock the Surface as pixels are copied.

    :note: Not compatible with numarray; you must use numpy or Numeric.

    :Parameters:
         `surface` : `Surface`
            Surface to copy.

    :rtype: Numeric or numpy array
    '''  
    array = array2d(surface)
    _array = pygame.array._array

    if surface._surf.flags & SDL_SRCCOLORKEY:
        # XXX No work with numarray
        colorkey = surface._surf.format.colorkey
        array = _array.choose(_array.equal(array, colorkey), (0, 0xff))
    else:
        array[:,:] = 0xff

    return array.astype(_array.UInt8)

def make_surface(array):
    '''Copy an array to a new surface.

    Create a new Surface that best resembles the data and format on the array.
    
    2D arrays are assumed to be 8-bit palette images, however no palette
    will be set.

    3D arrays are assumed to have the usual RGB components as the minor axis.

    :Parameters:
        `array` : Numeric array
            Image data.

    :rtype: `Surface`
    '''
    pygame.array._check_array()
    
    module = pygame.array._get_array_local_module(array)
    shape = module.shape(array)

    if len(shape) == 2:
        depth = 8
        Rmask = Gmask = Bmask = 0
    elif len(shape) == 3 and shape[2] == 3:
        depth = 32
        Rmask = 0xff << 16
        Gmask = 0xff << 8
        Bmask = 0xff
    else:
        raise ValueError, 'must be valid 2d or 3d array\n'
    
    surf = SDL_CreateRGBSurface(0, shape[0], shape[1], depth, 
                                Rmask, Gmask, Bmask, 0)
    surface = pygame.surface.Surface(surf=surf)
    blit_array(surface, array)
    return surface

def blit_array(surface, array):
    '''Blit directly from the values in an array.

    Directly copy values from an array into a Surface. This is faster than
    converting the array into a Surface and blitting. The array must be the
    same dimensions as the Surface and will completely replace all pixel
    values.

    2D arrays must have the same pixel format as the surface.

    This function will temporarily lock the Surface as the new values are
    copied.

    :Parameters:
        `surface` : `Surface`
            Surface to blit to.
        `array` : numpy, numarray or Numeric 2D or 3D array
            Image data.

    :rtype: `Surface`
    '''
    pygame.array._check_array()

    surf = surface._surf
    bpp = surf.format.BytesPerPixel

    # Local array module, may be different to global array module.
    module = pygame.array._get_array_local_module(array)

    # Transpose to traditional row ordering (row, column, [component])
    shape = module.shape(array)
    if len(shape) == 3 and shape[2] == 3:
        array = module.transpose(array, (1, 0, 2))
        f = surf.format
        array = (array[:,:,::3] >> f.Rloss << f.Rshift) | \
                (array[:,:,1::3] >> f.Gloss << f.Gshift) | \
                (array[:,:,2::3] >> f.Bloss << f.Bshift)
    elif len(shape) == 2:
        array = module.transpose(array)
    else:
        raise ValueError, 'must be a valid 2d or 3d array\n'
    shape = module.shape(array)

    if bpp <= 0 or bpp > 4:
        raise ValueError, 'unsupport bit depth for surface'

    if surf.w != shape[1] or surf.h != shape[0]:
        raise ValueError, 'array must match surface dimensions'

    if callable(array.itemsize):
        # numarray, Numeric
        itemsize = array.itemsize()
    else:
        # numpy
        itemsize = array.itemsize

    data = array.tostring()

    if itemsize > bpp:
        # Trim bytes from each element, keep least significant byte(s)
        if SDL_BYTEORDER == SDL_LIL_ENDIAN:
            pattern = '(%s)%s' % ('.' * bpp, '.' * (itemsize - bpp))
        else:
            pattern = '%s(%s)' % ('.' * (itemsize - bpp), '.' * bpp)
        data = ''.join(re.compile(pattern, flags=re.DOTALL).findall(data))
    elif itemsize < bpp:
        # Add pad bytes to each element, at most significant end
        pad = '\0' * (bpp - itemsize)
        pixels = re.compile('.' * itemsize, flags=re.DOTALL).findall(data)
        data = pad.join(pixels)
        if SDL_BYTEORDER == SDL_LIL_ENDIAN:
            data = data + pad
        else:
            data = pad + data

    # Add zeros pad for pitch correction
    pitchdiff = surf.pitch - surf.w * bpp 
    if pitchdiff > 0:
        pad = '\0' * pitchdiff
        rows = re.compile('.' * surf.w * bpp, flags=re.DOTALL).findall(data)
        data = pad.join(rows) + pad

    # Good to go
    surface.lock()
    memmove(surf.pixels.ptr, data, len(data))
    surface.unlock()

def map_array(surface, array):
    '''Map a 3d array into a 2d array.

    Convert a 3D array into a 2D array. This will use the given Surface format
    to control the conversion.  Palette surface formats are not supported.

    :note: Arrays need not be 3D, so long as the minor axis has three elements
        giving the component colours, any array shape can be used (for
        example, a single colour can be mapped, or an array of colours).

    :Parameters:
        `surface` : `Surface`
            Surface with format information.
        `array` : Numeric, numpy or numarray array
            Array to convert.

    :rtype: Numeric, numpy or numarray array
    :return: array module will be the same as array passed in.
    '''
    surf = surface._surf

    module = pygame.array._get_array_local_module(array)
    shape = module.shape(array)

    if shape[-1] != 3:
        # XXX Misleading: lists and single values also accepted
        raise ValueError, 'array must be a 3d array of 3-value color data'

    if surf.format.BytesPerPixel <= 0 or surf.format.BytesPerPixel > 4:
        raise ValueError, 'unsupport bit depth for surface array'

    f = surf.format
    array = (array[...,::3] >> f.Rloss << f.Rshift) | \
            (array[...,1::3] >> f.Gloss << f.Gshift) | \
            (array[...,2::3] >> f.Bloss << f.Bshift)

    return array


