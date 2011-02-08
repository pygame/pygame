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
##    Marcus von Appen
##    mva@sysfault.org

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

import pygame
from pygame.compat import bytes_
from pygame._arraysurfarray import blit_array
import numpy
from numpy import array as numpy_array

def array2d (surface):
    """pygame.numpyarray.array2d (Surface): return array

    copy pixels into a 2d array

    Copy the pixels from a Surface into a 2D array. The bit depth of the
    surface will control the size of the integer values, and will work
    for any type of pixel format.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    bpp = surface.get_bytesize ()
    if bpp <= 0 or bpp > 4:
        raise ValueError("unsupported bit depth for 2D array")

    size = surface.get_size ()
    width, height = size
    
    # Taken from Alex Holkner's pygame-ctypes package. Thanks a lot.
    data = numpy.frombuffer (surface.get_buffer (), numpy.uint8)
    pitch = surface.get_pitch ()
    row_size = width * bpp
    if pitch != row_size:
        data.shape = (height, pitch)
        data = data[:, 0:row_size]

    dtype = (None, numpy.uint8, numpy.uint16, numpy.int32, numpy.int32)[bpp]
    array = numpy.zeros (size, dtype, 'F')
    array_data = numpy.frombuffer (array, numpy.uint8)
    if bpp == 3:
        data.shape = (height, width, 3)
        array_data.shape = (height, width, 4)
        array_data[:,:,:3] = data[...]
    else:
        data.shape = (height, row_size)
        array_data.shape = (height, row_size)
        array_data[...] =  data[...]
    return array
    
def pixels2d (surface):
    """pygame.numpyarray.pixels2d (Surface): return array

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

    return numpy_array(surface.get_view('2'), copy=False)

def array3d (surface):
    """pygame.numpyarray.array3d (Surface): return array

    copy pixels into a 3d array

    Copy the pixels from a Surface into a 3D array. The bit depth of the
    surface will control the size of the integer values, and will work
    for any type of pixel format.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    bpp = surface.get_bytesize ()
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
        array = numpy.array (planes, numpy.uint8)
        array = numpy.transpose (array, (1, 2, 0))
        return array
    elif bpp == 2:
        # Taken from SDL_GetRGBA.
        masks = surface.get_masks ()
        shifts = surface.get_shifts ()
        losses = surface.get_losses ()
        vr = (array & masks[0]) >> shifts[0]
        vg = (array & masks[1]) >> shifts[1]
        vb = (array & masks[2]) >> shifts[2]
        planes = [(vr << losses[0]) + (vr >> (8 - (losses[0] << 1))),
                  (vg << losses[1]) + (vg >> (8 - (losses[1] << 1))),
                  (vb << losses[2]) + (vb >> (8 - (losses[2] << 1)))]
        array = numpy.array (planes, numpy.uint8)
        return numpy.transpose (array, (1, 2, 0))
    else:
        masks = surface.get_masks ()
        shifts = surface.get_shifts ()
        losses = surface.get_losses ()
        planes = [((array & masks[0]) >> shifts[0]), # << losses[0], Assume 0
                  ((array & masks[1]) >> shifts[1]), # << losses[1],
                  ((array & masks[2]) >> shifts[2])] # << losses[2]]
        array = numpy.array (planes, numpy.uint8)
        return numpy.transpose (array, (1, 2, 0))

def pixels3d (surface):
    """pygame.numpyarray.pixels3d (Surface): return array

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

    return numpy_array(surface.get_view('3'), copy=False)

def array_alpha (surface):
    """pygame.numpyarray.array_alpha (Surface): return array

    copy pixel alphas into a 2d array

    Copy the pixel alpha values (degree of transparency) from a Surface
    into a 2D array. This will work for any type of Surface
    format. Surfaces without a pixel alpha will return an array with all
    opaque values.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    if (surface.get_bytesize () == 1 or
        surface.get_alpha () is None or
        surface.get_masks ()[3] == 0):
        # 1 bpp surfaces and surfaces without per-pixel alpha are always
        # fully opaque.
        array = numpy.empty (surface.get_width () * surface.get_height (),
                             numpy.uint8)
        array.fill (0xff)
        array.shape = surface.get_width (), surface.get_height ()
        return array

    array = array2d (surface)
    if surface.get_bytesize () == 2:
        # Taken from SDL_GetRGBA.
        va = (array & surface.get_masks ()[3]) >> surface.get_shifts ()[3]
        array = ((va << surface.get_losses ()[3]) +
                 (va >> (8 - (surface.get_losses ()[3] << 1))))
    else:
        # Taken from _numericsurfarray.c.
        array = array >> surface.get_shifts ()[3] << surface.get_losses ()[3]
    array = array.astype (numpy.uint8)
    return array

def pixels_alpha (surface):
    """pygame.numpyarray.pixels_alpha (Surface): return array

    reference pixel alphas into a 2d array

    Create a new 2D array that directly references the alpha values
    (degree of transparency) in a Surface. Any changes to the array will
    affect the pixels in the Surface. This is a fast operation since no
    data is copied.

    This can only work on 32-bit Surfaces with a per-pixel alpha value.

    The Surface this array references will remain locked for the
    lifetime of the array.
    """

    return numpy.array (surface.get_view('a'), copy=False)

def pixels_red (surface):
    """pygame.surfarray.pixels_red (Surface): return array

    Reference pixel red into a 2d array.

    Create a new 2D array that directly references the red values
    in a Surface. Any changes to the array will affect the pixels
    in the Surface. This is a fast operation since no data is copied.

    This can only work on 24-bit or 32-bit Surfaces.

    The Surface this array references will remain locked for the
    lifetime of the array.
    """

    return numpy.array (surface.get_view('r'), copy=False)

def pixels_green (surface):
    """pygame.surfarray.pixels_green (Surface): return array

    Reference pixel green into a 2d array.

    Create a new 2D array that directly references the green values
    in a Surface. Any changes to the array will affect the pixels
    in the Surface. This is a fast operation since no data is copied.

    This can only work on 24-bit or 32-bit Surfaces.

    The Surface this array references will remain locked for the
    lifetime of the array.
    """

    return numpy.array (surface.get_view('g'), copy=False)

def pixels_blue (surface):
    """pygame.surfarray.pixels_blue (Surface): return array

    Reference pixel blue into a 2d array.

    Create a new 2D array that directly references the blue values
    in a Surface. Any changes to the array will affect the pixels
    in the Surface. This is a fast operation since no data is copied.

    This can only work on 24-bit or 32-bit Surfaces.

    The Surface this array references will remain locked for the
    lifetime of the array.
    """

    return numpy.array (surface.get_view('b'), copy=False)

def array_colorkey (surface):
    """pygame.numpyarray.array_colorkey (Surface): return array

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
        array = numpy.empty (surface.get_width () * surface.get_height (),
                             numpy.uint8)
        array.fill (0xff)
        array.shape = surface.get_width (), surface.get_height ()
        return array

    # Taken from from Alex Holkner's pygame-ctypes package. Thanks a
    # lot.
    array = array2d (surface)
    # Check each pixel value for the colorkey and mark it as opaque or
    # transparent as needed.
    val = surface.map_rgb (colorkey)
    array = numpy.choose (numpy.equal (array, val),
                          (numpy.uint8 (0xff), numpy.uint8 (0)))
    array.shape = surface.get_width (), surface.get_height ()
    return array

def make_surface (array):
    """pygame.numpyarray.make_surface (array): return Surface

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
        raise ValueError("must be a valid 2d or 3d array")

    surface = pygame.Surface ((shape[0], shape[1]), 0, bpp, (r, g, b, 0))
    blit_array (surface, array)
    return surface
    
def map_array (surface, array):
    """pygame.numpyarray.map_array (Surface, array3d): return array2d

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
    bpp = surface.get_bytesize ()
    if bpp <= 1 or bpp > 4:
        raise ValueError("unsupported bit depth for surface array")

    shape = array.shape
    if shape[-1] != 3:
        raise ValueError("array must be a 3d array of 3-value color data")

    shifts = surface.get_shifts ()
    losses = surface.get_losses ()
    if array.dtype != numpy.int32:
        array = array.astype(numpy.int32)
    out       = array[...,0] >> losses[0] << shifts[0]
    out[...] |= array[...,1] >> losses[1] << shifts[1]
    out[...] |= array[...,2] >> losses[2] << shifts[2]
    if surface.get_flags() & pygame.SRCALPHA:
        out[...] |= numpy.int32(255) >> losses[3] << shifts[3]
    return out
