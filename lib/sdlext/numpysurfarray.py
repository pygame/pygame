##    pygame - Python Game Library
##    Copyright (C) 2007 Marcus von Appen
##
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##

"""pygame module for accessing surface pixel data using numpy

Functions to convert pixel data between pygame Surfaces and Numpy
arrays. This module will only be available when pygame can use the
external Numpy package.

Note, that numpyarray is an optional module. It requires that Numpy is
installed to be used. If not installed, an exception will be raised when
it is used. eg. ImportError: no module named numpy

Every pixel is stored as a single integer value to represent the red,
green, and blue colors. The 8bit images use a value that looks into a
colormap. Pixels with higher depth use a bit packing process to place
three or four values into a single number.

The Numpy arrays are indexed by the X axis first, followed by the Y
axis. Arrays that treat the pixels as a single integer are referred to
as 2D arrays. This module can also separate the red, green, and blue
color values into separate indices. These types of arrays are referred
to as 3D arrays, and the last index is 0 for red, 1 for green, and 2 for
blue.

In contrast to Numeric Numpy does use unsigned 16bit integers, images
with 16bit data will be treated as unsigned integers.
"""

import pygame2.compat
pygame2.compat.deprecation ("""The numpysurfarray package is deprecated and
will be changed or removed in future versions""")

import pygame2.sdl.video as video
import pygame2.sdl.constants as sdlconst
import numpy
import re

def array2d (surface):
    """pygame2.sdlext.numpyarray.array2d (Surface) -> array

    copy pixels into a 2d array

    Copy the pixels from a Surface into a 2D array. The bit depth of the
    surface will control the size of the integer values, and will work
    for any type of pixel format.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    sformat = surface.format
    bpp = sformat.bytes_per_pixel
    if bpp <= 0 or bpp > 4:
        raise ValueError ("unsupported bit depth for 2D array")

    # Taken from Alex Holkner's pygame-ctypes package. Thanks a lot.
    data = surface.pixels
        
    # Remove extra pitch from each row.
    width = surface.width
    pitchdiff = sformat.pitch - width * bpp
    if pitchdiff > 0:
        pattern = re.compile ('(%s)%s' % ('.' * width * bpp, '.' * pitchdiff),
                              flags=re.DOTALL)
        data = ''.join (pattern.findall (data))

    if bpp == 3:
        # Pad each triplet of bytes with another zero
        pattern = re.compile ('...', flags=re.DOTALL)
        data = '\0'.join (pattern.findall (data))
        if sdlconst.BYTEORDER == sdlconst.LIL_ENDIAN:
            data += '\0'
        else:
            data = '\0' + data
        bpp = 4

    typecode = (numpy.uint8, numpy.uint16, None, numpy.uint32)[bpp - 1]
    array = numpy.fromstring (data, typecode)
    array.shape = (surface.get_height (), width)
    array = numpy.transpose (array)
    return array

def pixels2d (surface):
    """pygame2.sdlext.numpyarray.pixels2d (Surface) -> array

    reference pixels into a 2d array
    
    Create a new 2D array that directly references the pixel values in a
    Surface. Any changes to the array will affect the pixels in the
    Surface. This is a fast operation since no data is copied.

    Pixels from a 24-bit Surface cannot be referenced, but all other
    Surface bit depths can.

    The Surface this references will remain locked for the lifetime of
    the array (see the Surface.lock - lock the Surface memory for pixel
    access method).
    """
    sformat = surface.format
    bpp = sformat.bytes_per_pixel
    if bpp == 3 or bpp < 1 or bpp > 4:
        raise ValueError ("unsupported bit depth for 2D reference array")

    typecode = (numpy.uint8, numpy.uint16, None, numpy.uint32)[bpp - 1]
    array = numpy.frombuffer (surface.pixels, typecode)
    array.shape = surface.height, sformat.pitch / bpp

    # Padding correction for certain depth due to padding bytes.
    array = array[:, :surface.width]
    array = numpy.transpose (array)
    return array

def array3d (surface):
    """pygame2.sdlext.numpyarray.array3d (Surface) -> array

    copy pixels into a 3d array

    Copy the pixels from a Surface into a 3D array. The bit depth of the
    surface will control the size of the integer values, and will work
    for any type of pixel format.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    sformat = surface.format
    bpp = sformat.bytes_per_pixel
    array = array2d (surface)

    # Taken from from Alex Holkner's pygame-ctypes package. Thanks a
    # lot.
    if bpp == 1:
        palette = surface.get_palette ()
        # Resolve the correct values using the color palette
        pal_r = numpy.array ([c[0] for c in palette])
        pal_g = numpy.array ([c[1] for c in palette])
        pal_b = numpy.array ([c[2] for c in palette])
        planes = [numpy.choose (array, pal_r),
                  numpy.choose (array, pal_g),
                  numpy.choose (array, pal_b)]
        array = numpy.array (planes)
        array = numpy.transpose (array, (1, 2, 0))
        return array
    else:
        masks = sformat.masks
        shifts = sformat.shifts
        losses = sformat.losses
        planes = [((array & masks[0]) >> shifts[0]) << losses[0],
                  ((array & masks[1]) >> shifts[1]) << losses[1],
                  ((array & masks[2]) >> shifts[2]) << losses[2]]
        array = numpy.array (planes)
        return numpy.transpose (array, (1, 2, 0))

def pixels3d (surface):
    """pygame2.sdlext.numpyarray.pixels3d (Surface) -> array

    reference pixels into a 3d array

    Create a new 3D array that directly references the pixel values in a
    Surface. Any changes to the array will affect the pixels in the
    Surface. This is a fast operation since no data is copied.

    This will only work on Surfaces that have 24-bit or 32-bit
    formats. Lower pixel formats cannot be referenced.

    The Surface this references will remain locked for the lifetime of
    the array (see the Surface.lock - lock the Surface memory for pixel
    access method).
    """
    sformat = surface.format
    bpp = sformat.bytes_per_pixel
    if bpp < 3 or bpp > 4:
        raise ValueError ("unsupported bit depth for 3D reference array")
    lilendian = sdlconst.BYTEORDER == sdlconst.LIL_ENDIAN

    start = 0
    step = 0

    # Check for RGB or BGR surface.
    shifts = sformat.shifts
    if shifts[0] == 16 and shifts[1] == 8 and shifts[2] == 0:
        # RGB 
        if lilendian:
            start = 2
            step = -1
        else:
            start = 0
            step = 1
    elif shifts[2] == 16 and shifts[1] == 8 and shifts[0] == 0:
        # BGR
        if lilendian:
            start = 0
            step = 1
        else:
            start = 2
            step = -1
    else:
        raise ValueError ("unsupported colormasks for 3D reference array")

    if bpp == 4 and not lilendian:
        start += 1

    array = numpy.ndarray \
            (shape=(surface.width, surface.height, 3),
             dtype=numpy.uint8, buffer=surface.pixels,
             offset=start, strides=(bpp, sformat.pitch, step))
    return array

def array_alpha (surface):
    """pygame2.sdlext.numpyarray.array_alpha (Surface) -> array

    copy pixel alphas into a 2d array

    Copy the pixel alpha values (degree of transparency) from a Surface
    into a 2D array. This will work for any type of Surface
    format. Surfaces without a pixel alpha will return an array with all
    opaque values.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    sformat = surface.format
    if sformat.bytes_per_pixel == 1 or not surface.get_alpha ():
        # 1 bpp surfaces and surfaces without alpha are always fully
        # opaque.
        array = numpy.empty (surface.width * surface.height, numpy.uint8)
        array.fill (0xff)
        array.shape = surface.width, surface.height
        return array

    array = array2d (surface)
    # Those shifts match the results from the old numeric system
    # exactly.
    array = array >> sformat.shifts[3] << sformat.losses[3]
    array = array.astype (numpy.uint8)
    return array

def pixels_alpha (surface):
    """pygame2.sdlext.numpyarray.pixels_alpha (Surface) -> array

    reference pixel alphas into a 2d array

    Create a new 2D array that directly references the alpha values
    (degree of transparency) in a Surface. Any changes to the array will
    affect the pixels in the Surface. This is a fast operation since no
    data is copied.

    This can only work on 32-bit Surfaces with a per-pixel alpha value.

    The Surface this array references will remain locked for the
    lifetime of the array.
    """
    sformat = surface.format
    if sformat.bytes_per_pixel != 4:
        raise ValueError ("unsupported bit depth for alpha reference array")
    lilendian = sdlconst.BYTEORDER == sdlconst.LIL_ENDIAN

    # ARGB surface.
    start = 0
    
    if sformat.shifts[3] == 24 and lilendian:
        # RGBA surface.
        start = 3
    elif sformat.shifts[3] == 0 and not lilendian:
        start = 3
    else:
        raise ValueError ("unsupported colormasks for alpha reference array")

    array = numpy.ndarray \
            (shape=(surface.width, surface.height),
             dtype=numpy.uint8, buffer=surface.pixels,
             offset=start, strides=(4, sformat.pitch))
    return array

def array_colorkey (surface):
    """pygame2.sdlext.numpyarray.array_colorkey (Surface) -> array

    copy the colorkey values into a 2d array

    Create a new array with the colorkey transparency value from each
    pixel. If the pixel matches the colorkey it will be fully
    tranparent; otherwise it will be fully opaque.

    This will work on any type of Surface format. If the image has no
    colorkey a solid opaque array will be returned.

    This function will temporarily lock the Surface as pixels are
    copied.
    """
    colorkey = surface.get_colorkey ()
    if colorkey == None:
        # No colorkey, return a solid opaque array.
        array = numpy.empty (surface.width * surface.height, numpy.uint8)
        array.fill (0xff)
        array.shape = surface.width, surface.height
        return array

    # Taken from from Alex Holkner's pygame-ctypes package. Thanks a
    # lot.
    array = array2d (surface)
    # Check each pixel value for the colorkey and mark it as opaque or
    # transparent as needed.
    val = surface.format.map_rgb (colorkey)
    array = numpy.choose (numpy.equal (array, val),
                          (numpy.uint8 (0xff), numpy.uint8 (0)))
    array.shape = surface.width, surface.height
    return array

def make_surface (array):
    """pygame2.sdlext.numpyarray.make_surface (array) -> Surface

    copy an array to a new surface

    Create a new Surface that best resembles the data and format on the
    array. The array can be 2D or 3D with any sized integer values.
    """ 
    # Taken from from Alex Holkner's pygame-ctypes package. Thanks a
    # lot.
    bpp = 0
    r = g = b = 0
    shape = array.shape
    if len (shape) == 2:
        # 2D array
        bpp = 8
        r = 0xFF >> 6 << 5
        g = 0xFF >> 5 << 2
        b = 0xFF >> 6
    elif len (shape) == 3 and shape[2] == 3:
        bpp = 32
        r = 0xff << 16
        g = 0xff << 8
        b = 0xff
    else:
        raise ValueError ("must be a valid 2d or 3d array")

    surface = video.Surface ((shape[0], shape[1]), 0, bpp, (r, g, b, 0))
    blit_array (surface, array)
    return surface

def blit_array (surface, array):
    """pygame2.sdlext.numpyarray.blit_array (Surface, array) -> None

    blit directly from a array values

    Directly copy values from an array into a Surface. This is faster
    than converting the array into a Surface and blitting. The array
    must be the same dimensions as the Surface and will completely
    replace all pixel values.

    This function will temporarily lock the Surface as the new values
    are copied.
    """
    sformat = surface.format
    bpp = sformat.bytes_per_pixel
    if bpp <= 0 or bpp > 4:
        raise ValueError ("unsupported bit depth for surface")
    
    shape = array.shape
    width = surface.width

    typecode = (numpy.uint8, numpy.uint16, None, numpy.uint32)[bpp - 1]
    array = array.astype (typecode)

    # Taken from from Alex Holkner's pygame-ctypes package. Thanks a
    # lot.
    if len(shape) == 3 and shape[2] == 3:
        array = numpy.transpose (array, (1, 0, 2))
        shifts = sformat.shifts
        losses = sformat.losses
        array = (array[:, :, ::3] >> losses[0] << shifts[0]) | \
                (array[:, :, 1::3] >> losses[1] << shifts[1]) | \
                (array[:, :, 2::3] >> losses[2] << shifts[2])
    elif len (shape) == 2:
        array = numpy.transpose (array)
    else:
        raise ValueError ("must be a valid 2d or 3d array")

    if width != shape[0] or surface.height != shape[1]:
        raise ValueError ("array must match the surface dimensions")

    itemsize = array.itemsize
    data = array.tostring ()

    if itemsize > bpp:
        # Trim bytes from each element, keep least significant byte(s)
        pattern = '%s(%s)' % ('.' * (itemsize - bpp), '.' * bpp)
        if sdlconst.BYTEORDER == sdlconst.LIL_ENDIAN:
            pattern = '(%s)%s' % ('.' * bpp, '.' * (itemsize - bpp))
        data = ''.join (re.compile (pattern, flags=re.DOTALL).findall (data))
    elif itemsize < bpp:
        # Add pad bytes to each element, at most significant end
        pad = '\0' * (bpp - itemsize)
        pixels = re.compile ('.' * itemsize, flags=re.DOTALL).findall (data)
        data = pad.join (pixels)
        if sdlconst.BYTEORDER == sdlconst.LIL_ENDIAN:
            data = data + pad
        else:
            data = pad + data

    # Add zeros pad for pitch correction
    pitchdiff = sformat.pitch - width * bpp
    if pitchdiff > 0:
        pad = '\0' * pitchdiff
        rows = re.compile ('.' * width * bpp, flags=re.DOTALL).findall (data)
        data = pad.join (rows) + pad

    surface.pixels.write (data, 0)
    
def map_array (surface, array):
    """pygame2.sdlext.numpyarray.map_array (Surface, array3d) -> array2d

    map a 3d array into a 2d array

    Convert a 3D array into a 2D array. This will use the given Surface
    format to control the conversion. Palette surface formats are not
    supported.

    Note: arrays do not need to be 3D, as long as the minor axis has
    three elements giving the component colours, any array shape can be
    used (for example, a single colour can be mapped, or an array of
    colours).
    """
    # Taken from from Alex Holkner's pygame-ctypes package. Thanks a
    # lot.
    sformat = surface.format
    bpp = sformat.bytes_per_pixel
    if bpp <= 0 or bpp > 4:
        raise ValueError ("unsupported bit depth for surface array")

    shape = array.shape
    if shape[-1] != 3:
        raise ValueError ("array must be a 3d array of 3-value color data")

    shifts = sformat.shifts
    losses = sformat.losses
    array = (array[:, :, ::3] >> losses[0] << shifts[0]) | \
            (array[:, :, 1::3] >> losses[1] << shifts[1]) | \
            (array[:, :, 2::3] >> losses[2] << shifts[2])
    return array
