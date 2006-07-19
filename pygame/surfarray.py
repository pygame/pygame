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
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from ctypes import *
import re

from SDL import *

# Numeric doesn't provide a Python interface to its array strides, which
# we need to modify, so we go under the bonnet and fiddle with it directly.
# This probably won't work in a lot of scenarios (i.e., outside of my 
# house).
class _PyArrayObject(Structure):
    _fields_ = [('ob_refcnt', c_int),
                ('ob_type', c_void_p),
                ('data', c_char_p),
                ('nd', c_int),
                ('dimensions', POINTER(c_int)),
                ('strides', POINTER(c_int)),
                ('base', c_void_p),
                ('descr', c_void_p),
                ('flags', c_uint),
                ('weakreflist', c_void_p)]

# Provide support for numpy and numarray in addition to Numeric.  To
# be compatible with Pygame, by default the module will be unavailable
# if Numeric is not available.  You can activate it to use any available
# array module by calling set_array_module().
try:
    import Numeric
    _array = Numeric
except ImportError:
    _array = None

def set_array_module(module=None):
    '''Set the array module to use; numpy, numarray or Numeric.

    If no arguments are given, every array module is tried and the
    first one that can be imported will be used.  The order of
    preference is numpy, numarray, Numeric.  You can determine which
    module was set with `get_array_module`.

    :Parameters:
        `module` : module or None
            Module to use.

    '''
    global _array
    if not module:
        for name in ('numpy', 'numarray', 'Numeric'):
            try:
                set_array_module(__import__(name, locals(), globals(), []))
            except ImportError:
                pass
    else:
        _array = module

def get_array_module():
    '''Get the currently set array module.

    If None is returned, no array module is set and the surfarray
    functions will not be useable.

    :rtype: module
    '''
    return _array

def _check_array():
    if not _array:
        raise ImportError, \
              'No array module set; use set_array_module if you want to ' + \
              'use numpy or numarray instead of Numeric.'

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
    _check_array()

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

    if bpp == 1:
        t = _array.UInt8
    elif bpp == 2:
        t = _array.UInt16
    elif bpp == 3:
        # Pad each triplet of bytes with another zero
        pattern = re.compile('...', flags=re.DOTALL)
        data = '\0'.join(pattern.findall(data))
        if SDL_BYTEORDER == SDL_LIL_ENDIAN:
            data += '\0'
        else:
            data = '\0' + data
        t = _array.UInt32
        bpp = 4
    elif bpp == 4:
        t = _array.UInt32

    shape = surf.h, surf.w

    if _array.__name__ == 'numpy':
        ar = _array.fromstring(data, t).reshape(shape)
    elif _array.__name__ == 'numarray':
        ar = _array.fromstring(data, t, shape)
    elif _array.__name__ == 'Numeric':
        ar = _array.fromstring(data, t).resize(shape)

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
            Surface to copy.

    :rtype: Numeric array
    '''       

def array3d(surface):
    '''Copy pixels into a 3d array.

    Copy the pixels from a Surface into a 3D array.  Arrays are indexed
    by [x,y,colour], where colour is 0 for the red component, 1 for the
    green component and 2 for the blue component.

    The pixel format of the surface will control the maximum size of the
    integer values.

    This function will temporarily lock the Surface as pixels are copied (see
    the Surface.lock() method).

    :Parameters:
         `surface` : `Surface`
            Surface to copy.

    :rtype: Numeric array
    '''
    array = array2d(surface)

    surf = surface._surf
    format = surf.format
    bpp = format.BytesPerPixel

    if format.BytesPerPixel == 1:
        raise NotImplementedException, 'TODO: palette lookup'
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
            Surface to copy.

    :rtype: Numeric array
    '''  

def array_alpha(surface):
    '''Copy pixel alphas into a 2d array.

    Copy the pixel alpha values (degree of transparency) from a Surface into a
    2D array. This will work for any type of Surface format. Surfaces without
    a pixel alpha will return an array with all opaque values.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock() method).

    :Parameters:
         `surface` : `Surface`
            Surface to copy.

    :rtype: Numeric array
    '''  

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
            Surface to copy.

    :rtype: Numeric array
    '''  

def array_colorkey(surface):
    '''Copy the colorkey values into a 2d array.

    Create a new array with the colorkey transparency value from each pixel. If
    the pixel matches the colorkey it will be fully tranparent; otherwise it
    will be fully opaque.

    This will work on any type of Surface format. If the image has no colorkey
    a solid opaque array will be returned.

    This function will temporarily lock the Surface as pixels are copied.

    :Parameters:
         `surface` : `Surface`
            Surface to copy.

    :rtype: Numeric array
    '''  

def make_surface(array):
    '''Copy an array to a new surface.

    Create a new Surface that best resembles the data and format on the array.
    The array can be 2D or 3D with any sized integer values.

    :Parameters:
        `array` : Numeric array
            Image data.

    :rtype: `Surface`
    '''

def _get_array_module(array):
    # Given an array, determine what array module it is from.  Note that
    # we don't require it to be the same module as _array, which is
    # only for returned arrays.

    # "strides" attribute is different in each module, so is hacky way
    # to check.
    if hasattr(array, 'strides'):
        import numpy
        return numpy
    elif hasattr(array, '_strides'):
        import numarray
        return numarray
    else:
        import Numeric
        return Numeric

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
    _check_array()

    surf = surface._surf
    bpp = surf.format.BytesPerPixel

    # Local array module, may be different to global array module.
    module = _get_array_module(array)

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

    print itemsize, bpp
    if itemsize > bpp:
        print 'a'
        # Trim bytes from each element, keep least significant byte(s)
        if SDL_BYTEORDER == SDL_LIL_ENDIAN:
            pattern = '(%s)%s' % ('.' * bpp, '.' * (itemsize - bpp))
        else:
            pattern = '%s(%s)' % ('.' * (itemsize - bpp), '.' * bpp)
        data = ''.join(re.compile(pattern, flags=re.DOTALL).findall(data))
    elif itemsize < bpp:
        print 'b'
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
        print 'c'
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
    to control the conversion.

    :Parameters:
        `surface` : `Surface`
            Surface with format information.
        `array` : Numeric array
            Array to convert.

    :rtype: Numeric array
    '''
