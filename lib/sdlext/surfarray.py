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

"""pygame2 module for accessing surface pixel data using array interfaces

Functions to convert pixel data between pygame Surfaces and arrays. This
module will only be functional when pygame can use the external Numpy or
Numeric packages.

Every pixel is stored as a single integer value to represent the red,
green, and blue colors. The 8bit images use a value that looks into a
colormap. Pixels with higher depth use a bit packing process to place
three or four values into a single number.

The arrays are indexed by the X axis first, followed by the Y
axis. Arrays that treat the pixels as a single integer are referred to
as 2D arrays. This module can also separate the red, green, and blue
color values into separate indices. These types of arrays are referred
to as 3D arrays, and the last index is 0 for red, 1 for green, and 2 for
blue.

Supported array types are

  numeric
  numpy

The default will be Numeric, if installed. Otherwise, numpy will be set
as default if installed. If neither Numeric nor numpy are installed, the
module will raise an ImportError.

The array type to use can be changed at runtime using the use_arraytype()
method, which requires one of the above types as string.

Note: numpy and Numeric are not completely compatible. Certain array
manipulations, which work for one type, might behave differently or even
completely break for the other.

Additionally, in contrast to Numeric numpy does use unsigned 16-bit
integers. Images with 16-bit data will be treated as unsigned
integers. Numeric instead uses signed integers for the representation,
which is important to keep in mind, if you use the module's functions
and wonder about the values.
"""

import pygame2.compat
pygame2.compat.deprecation ("""The surfarray package is deprecated and
will be changed or removed in future versions""")

# Global array type setting. See use_arraytype().
__arraytype = None

# Try to import the necessary modules.
try:
    import pygame2.sdlext.numericsurfarray as numericsf
    __hasnumeric = True
    __arraytype = "numeric"
except ImportError:
    __hasnumeric = False

try:
    import pygame2.sdlext.numpysurfarray as numpysf
    __hasnumpy = True
    if not __hasnumeric:
        __arraytype = "numpy"
except ImportError:
    __hasnumpy = False

#if not __hasnumpy and not __hasnumeric:
#    raise ImportError ("no module named numpy or Numeric found")

def array2d (surface):
    """pygame2.sdlext.surfarray.array2d (Surface) -> array

    Copy pixels into a 2d array.

    Copy the pixels from a Surface into a 2D array. The bit depth of the
    surface will control the size of the integer values, and will work
    for any type of pixel format.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    if __arraytype == "numeric":
        return numericsf.array2d (surface)
    elif __arraytype == "numpy":
        return numpysf.array2d (surface)
    raise NotImplementedError ("surface arrays are not supported")

def pixels2d (surface):
    """pygame2.sdlext.surfarray.pixels2d (Surface) -> array

    Reference pixels into a 2d array.
    
    Create a new 2D array that directly references the pixel values in a
    Surface. Any changes to the array will affect the pixels in the
    Surface. This is a fast operation since no data is copied.

    Pixels from a 24-bit Surface cannot be referenced, but all other
    Surface bit depths can.

    The Surface this references will remain locked for the lifetime of
    the array (see the Surface.lock - lock the Surface memory for pixel
    access method).
    """
    if __arraytype == "numeric":
        return numericsf.pixels2d (surface)
    elif __arraytype == "numpy":
        return numpysf.pixels2d (surface)
    raise NotImplementedError ("surface arrays are not supported")

def array3d (surface):
    """pygame2.sdlext.surfarray.array3d (Surface) -> array

    Copy pixels into a 3d array.

    Copy the pixels from a Surface into a 3D array. The bit depth of the
    surface will control the size of the integer values, and will work
    for any type of pixel format.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    if __arraytype == "numeric":
        return numericsf.array3d (surface)
    elif __arraytype == "numpy":
        return numpysf.array3d (surface)
    raise NotImplementedError ("surface arrays are not supported")

def pixels3d (surface):
    """pygame2.sdlext.surfarray.pixels3d (Surface) -> array

    Reference pixels into a 3d array.

    Create a new 3D array that directly references the pixel values in a
    Surface. Any changes to the array will affect the pixels in the
    Surface. This is a fast operation since no data is copied.

    This will only work on Surfaces that have 24-bit or 32-bit
    formats. Lower pixel formats cannot be referenced.

    The Surface this references will remain locked for the lifetime of
    the array (see the Surface.lock - lock the Surface memory for pixel
    access method).
    """
    if __arraytype == "numeric":
        return numericsf.pixels3d (surface)
    elif __arraytype == "numpy":
        return numpysf.pixels3d (surface)
    raise NotImplementedError ("surface arrays are not supported")

def array_alpha (surface):
    """pygame2.sdlext.surfarray.array_alpha (Surface) -> array

    Copy pixel alphas into a 2d array.

    Copy the pixel alpha values (degree of transparency) from a Surface
    into a 2D array. This will work for any type of Surface
    format. Surfaces without a pixel alpha will return an array with all
    opaque values.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    if __arraytype == "numeric":
        return numericsf.array_alpha (surface)
    elif __arraytype == "numpy":
        return numpysf.array_alpha (surface)
    raise NotImplementedError ("surface arrays are not supported")

def pixels_alpha (surface):
    """pygame2.sdlext.surfarray.pixels_alpha (Surface) -> array

    Reference pixel alphas into a 2d array.

    Create a new 2D array that directly references the alpha values
    (degree of transparency) in a Surface. Any changes to the array will
    affect the pixels in the Surface. This is a fast operation since no
    data is copied.

    This can only work on 32-bit Surfaces with a per-pixel alpha value.

    The Surface this array references will remain locked for the
    lifetime of the array.
    """
    if __arraytype == "numeric":
        return numericsf.pixels_alpha (surface)
    elif __arraytype == "numpy":
        return numpysf.pixels_alpha (surface)
    raise NotImplementedError  ("surface arrays are not supported")

def array_colorkey (surface):
    """pygame2.sdlext.surfarray.array_colorkey (Surface) -> array

    Copy the colorkey values into a 2d array.

    Create a new array with the colorkey transparency value from each
    pixel. If the pixel matches the colorkey it will be fully
    transparent; otherwise it will be fully opaque.

    This will work on any type of Surface format. If the image has no
    colorkey a solid opaque array will be returned.

    This function will temporarily lock the Surface as pixels are
    copied.
    """
    if __arraytype == "numeric":
        return numericsf.array_colorkey (surface)
    elif __arraytype == "numpy":
        return numpysf.array_colorkey (surface)
    raise NotImplementedError ("surface arrays are not supported")

def make_surface (array):
    """pygame2.sdlext.surfarray.make_surface (array) -> Surface

    Copy an array to a new surface.

    Create a new Surface that best resembles the data and format on the
    array. The array can be 2D or 3D with any sized integer values.
    """ 
    if __arraytype == "numeric":
        return numericsf.make_surface (array)
    elif __arraytype == "numpy":
        return numpysf.make_surface (array)
    raise NotImplementedError ("surface arrays are not supported")

def blit_array (surface, array):
    """pygame2.sdlext.surfarray.blit_array (Surface, array) -> None

    Blit directly from a array values.

    Directly copy values from an array into a Surface. This is faster
    than converting the array into a Surface and blitting. The array
    must be the same dimensions as the Surface and will completely
    replace all pixel values.

    This function will temporarily lock the Surface as the new values
    are copied.
    """
    if __arraytype == "numeric":
        return numericsf.blit_array (surface, array)
    elif __arraytype == "numpy":
        return numpysf.blit_array (surface, array)
    raise NotImplementedError ("surface arrays are not supported")

def map_array (surface, array):
    """pygame2.sdlext.surfarray.map_array (Surface, array3d) -> array2d

    Map a 3D array into a 2D array.

    Convert a 3D array into a 2D array. This will use the given Surface
    format to control the conversion. Palette surface formats are not
    supported.
    """
    if __arraytype == "numeric":
        return numericsf.map_array (surface, array)
    elif __arraytype == "numpy":
        return numpysf.map_array (surface, array)
    raise NotImplementedError ("surface arrays are not supported")

def use_arraytype (arraytype):
    """pygame2.sdlext.surfarray.use_arraytype (arraytype) -> None

    Sets the array system to be used for surface arrays.

    Uses the requested array type for the module functions.
    Currently supported array types are:

      numeric 
      numpy

    If the requested type is not available, a ValueError will be raised.
    """
    global __arraytype

    arraytype = arraytype.lower ()
    if arraytype == "numeric":
        if __hasnumeric:
            __arraytype = arraytype
        else:
            raise ValueError ("Numeric arrays are not available")
        
    elif arraytype == "numpy":
        if __hasnumpy:
            __arraytype = arraytype
        else:
            raise ValueError ("numpy arrays are not available")
    else:
        raise ValueError ("invalid array type")

def get_arraytype ():
    """pygame2.sdlext.surfarray.get_arraytype () -> str

    Gets the currently active array type.

    Returns the currently active array type. This will be a value of the
    get_arraytypes() tuple and indicates which type of array module is
    used for the array creation.
    """
    return __arraytype

def get_arraytypes ():
    """pygame2.sdlext.surfarray.get_arraytypes () -> tuple

    Gets the array system types currently supported.

    Checks, which array system types are available and returns them as a
    tuple of strings. The values of the tuple can be used directly in
    the use_arraytype () method.

    If no supported array system could be found, None will be returned.
    """
    vals = []
    if __hasnumeric:
        vals.append ("numeric")
    if __hasnumpy:
        vals.append ("numpy")
    if len (vals) == 0:
        return None
    return tuple (vals)
