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

"""pygame module for accessing surface pixel data using array interfaces

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

  numpy
  numeric (deprecated; will be removed in Pygame 1.9.3.)

The default will be numpy, if installed. Otherwise, Numeric will be set
as default if installed, and a deprecation warning will be issued. If
neither numpy nor Numeric are installed, the module will raise an
ImportError.

The array type to use can be changed at runtime using the use_arraytype()
method, which requires one of the above types as string.

Note: numpy and Numeric are not completely compatible. Certain array
manipulations, which work for one type, might behave differently or even
completely break for the other.

Additionally, in contrast to Numeric, numpy does use unsigned 16-bit
integers. Images with 16-bit data will be treated as unsigned
integers. Numeric instead uses signed integers for the representation,
which is important to keep in mind, if you use the module's functions
and wonder about the values.
"""

# Try to import the necessary modules.
# import pygame._numpysurfarray as numpysf
numpysf = None


from pygame.pixelcopy import array_to_surface, make_surface as pc_make_surface

__all__ = ["array_to_surface", "pc_make_surface"]

def blit_array(surface, array):
    """pygame.surfarray.blit_array(Surface, array): return None

    Blit directly from a array values.

    Directly copy values from an array into a Surface. This is faster than
    converting the array into a Surface and blitting. The array must be the
    same dimensions as the Surface and will completely replace all pixel
    values. Only integer, ascii character and record arrays are accepted.

    This function will temporarily lock the Surface as the new values are
    copied.
    """
    global numpysf
    try:
        return numpysf.blit_array(surface, array)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.blit_array(surface, array)


def array2d(surface):
    """pygame.surfarray.array2d(Surface): return array

    Copy pixels into a 2d array.

    Copy the pixels from a Surface into a 2D array. The bit depth of the
    surface will control the size of the integer values, and will work
    for any type of pixel format.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    global numpysf
    try:
        return numpysf.array2d(surface)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.array2d(surface)


def pixels2d(surface):
    """pygame.surfarray.pixels2d(Surface): return array

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
    global numpysf
    try:
        return numpysf.pixels2d(surface)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.pixels2d(surface)


def array3d(surface):
    """pygame.surfarray.array3d(Surface): return array

    Copy pixels into a 3d array.

    Copy the pixels from a Surface into a 3D array. The bit depth of the
    surface will control the size of the integer values, and will work
    for any type of pixel format.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    global numpysf
    try:
        return numpysf.array3d(surface)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.array3d(surface)


def pixels3d(surface):
    """pygame.surfarray.pixels3d(Surface): return array

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
    global numpysf
    try:
        return numpysf.pixels3d(surface)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.pixels3d(surface)


def array_alpha(surface):
    """pygame.surfarray.array_alpha(Surface): return array

    Copy pixel alphas into a 2d array.

    Copy the pixel alpha values (degree of transparency) from a Surface
    into a 2D array. This will work for any type of Surface
    format. Surfaces without a pixel alpha will return an array with all
    opaque values.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    global numpysf
    try:
        return numpysf.array_alpha(surface)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.array_alpha(surface)


def pixels_alpha(surface):
    """pygame.surfarray.pixels_alpha(Surface): return array

    Reference pixel alphas into a 2d array.

    Create a new 2D array that directly references the alpha values
    (degree of transparency) in a Surface. Any changes to the array will
    affect the pixels in the Surface. This is a fast operation since no
    data is copied.

    This can only work on 32-bit Surfaces with a per-pixel alpha value.

    The Surface this array references will remain locked for the
    lifetime of the array.
    """
    global numpysf
    try:
        return numpysf.pixels_alpha(surface)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.pixels_alpha(surface)


def pixels_red(surface):
    """pygame.surfarray.pixels_red(Surface): return array

    Reference pixel red into a 2d array.

    Create a new 2D array that directly references the red values
    in a Surface. Any changes to the array will affect the pixels
    in the Surface. This is a fast operation since no data is copied.

    This can only work on 24-bit or 32-bit Surfaces.

    The Surface this array references will remain locked for the
    lifetime of the array.
    """
    global numpysf
    try:
        return numpysf.pixels_red(surface)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.pixels_red(surface)


def pixels_green(surface):
    """pygame.surfarray.pixels_green(Surface): return array

    Reference pixel green into a 2d array.

    Create a new 2D array that directly references the green values
    in a Surface. Any changes to the array will affect the pixels
    in the Surface. This is a fast operation since no data is copied.

    This can only work on 24-bit or 32-bit Surfaces.

    The Surface this array references will remain locked for the
    lifetime of the array.
    """
    global numpysf
    try:
        return numpysf.pixels_green(surface)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.pixels_green(surface)


def pixels_blue(surface):
    """pygame.surfarray.pixels_blue(Surface): return array

    Reference pixel blue into a 2d array.

    Create a new 2D array that directly references the blue values
    in a Surface. Any changes to the array will affect the pixels
    in the Surface. This is a fast operation since no data is copied.

    This can only work on 24-bit or 32-bit Surfaces.

    The Surface this array references will remain locked for the
    lifetime of the array.
    """
    global numpysf
    try:
        return numpysf.pixels_blue(surface)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.pixels_blue(surface)


def array_colorkey(surface):
    """pygame.surfarray.array_colorkey(Surface): return array

    Copy the colorkey values into a 2d array.

    Create a new array with the colorkey transparency value from each
    pixel. If the pixel matches the colorkey it will be fully
    tranparent; otherwise it will be fully opaque.

    This will work on any type of Surface format. If the image has no
    colorkey a solid opaque array will be returned.

    This function will temporarily lock the Surface as pixels are
    copied.
    """
    global numpysf
    try:
        return numpysf.array_colorkey(surface)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.array_colorkey(surface)


def make_surface(array):
    """pygame.surfarray.make_surface(array): return Surface

    Copy an array to a new surface.

    Create a new Surface that best resembles the data and format on the
    array. The array can be 2D or 3D with any sized integer values.
    """
    global numpysf
    try:
        return numpysf.make_surface(array)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.make_surface(array)


def map_array(surface, array):
    """pygame.surfarray.map_array(Surface, array3d): return array2d

    Map a 3D array into a 2D array.

    Convert a 3D array into a 2D array. This will use the given Surface
    format to control the conversion. Palette surface formats are not
    supported.
    """
    global numpysf
    try:
        return numpysf.map_array(surface, array)
    except AttributeError:
        import pygame._numpysurfarray as numpysf
        return numpysf.map_array(surface, array)


def use_arraytype(arraytype):
    """pygame.surfarray.use_arraytype(arraytype): return None

    DEPRECATED - only numpy arrays are now supported.
    """
    arraytype = arraytype.lower()
    if arraytype != "numpy":
        raise ValueError("invalid array type")

def get_arraytype():
    """pygame.surfarray.get_arraytype(): return str

    DEPRECATED - only numpy arrays are now supported.
    """
    return "numpy"

def get_arraytypes():
    """pygame.surfarray.get_arraytypes(): return tuple

    DEPRECATED - only numpy arrays are now supported.
    """
    return ("numpy",)
