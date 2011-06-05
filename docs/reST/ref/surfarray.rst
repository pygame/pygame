.. include:: common.txt

:mod:`pygame.surfarray`
=======================

.. module:: pygame.surfarray
   :synopsis: pygame module for accessing surface pixel data using array interfaces

| :sl:`pygame module for accessing surface pixel data using array interfaces`

Functions to convert pixel data between pygame Surfaces and arrays. This module
will only be functional when pygame can use the external Numpy or Numeric
packages.

Every pixel is stored as a single integer value to represent the red, green,
and blue colors. The 8bit images use a value that looks into a colormap. Pixels
with higher depth use a bit packing process to place three or four values into
a single number.

The arrays are indexed by the ``X`` axis first, followed by the ``Y`` axis.
Arrays that treat the pixels as a single integer are referred to as 2D arrays.
This module can also separate the red, green, and blue color values into
separate indices. These types of arrays are referred to as 3D arrays, and the
last index is 0 for red, 1 for green, and 2 for blue.

Supported array systems are

::

  numpy
  numeric (deprecated; to be removed in Pygame 1.9.3.)

The default will be numpy, if installed. Otherwise, Numeric will be set as
default if installed, and a deprecation warning will be issued. If neither
numpy nor Numeric are installed, the module will raise an ImportError.

The array type to use can be changed at runtime using the use_arraytype ()
method, which requires one of the above types as string.

Note: numpy and Numeric are not completely compatible. Certain array
manipulations, which work for one type, might behave differently or even
completely break for the other.

Additionally, in contrast to Numeric, numpy does use unsigned 16-bit integers.
Images with 16-bit data will be treated as unsigned integers. Numeric instead
uses signed integers for the representation, which is important to keep in
mind, if you use the module's functions and wonder about the values.

The support of numpy is new in Pygame 1.8. Official Numeric deprecation begins
in Pygame 1.9.2.

.. function:: array2d

   | :sl:`Copy pixels into a 2d array`
   | :sg:`array2d(Surface) -> array`

   Copy the pixels from a Surface into a 2D array. The bit depth of the surface
   will control the size of the integer values, and will work for any type of
   pixel format.

   This function will temporarily lock the Surface as pixels are copied (see
   the :func:`Surface.lock` - lock the Surface memory for pixel access method).

   .. ## pygame.surfarray.array2d ##

.. function:: pixels2d

   | :sl:`Reference pixels into a 2d array`
   | :sg:`pixels2d(Surface) -> array`

   Create a new 2D array that directly references the pixel values in a
   Surface. Any changes to the array will affect the pixels in the Surface.
   This is a fast operation since no data is copied.

   Pixels from a 24-bit Surface cannot be referenced, but all other Surface bit
   depths can.

   The Surface this references will remain locked for the lifetime of the array
   (see the :func:`Surface.lock` - lock the Surface memory for pixel access
   method).

   .. ## pygame.surfarray.pixels2d ##

.. function:: array3d

   | :sl:`Copy pixels into a 3d array`
   | :sg:`array3d(Surface) -> array`

   Copy the pixels from a Surface into a 3D array. The bit depth of the surface
   will control the size of the integer values, and will work for any type of
   pixel format.

   This function will temporarily lock the Surface as pixels are copied (see
   the :func:`Surface.lock` - lock the Surface memory for pixel access method).

   .. ## pygame.surfarray.array3d ##

.. function:: pixels3d

   | :sl:`Reference pixels into a 3d array`
   | :sg:`pixels3d(Surface) -> array`

   Create a new 3D array that directly references the pixel values in a
   Surface. Any changes to the array will affect the pixels in the Surface.
   This is a fast operation since no data is copied.

   This will only work on Surfaces that have 24-bit or 32-bit formats. Lower
   pixel formats cannot be referenced.

   The Surface this references will remain locked for the lifetime of the array
   (see the :func:`Surface.lock` - lock the Surface memory for pixel access
   method).

   .. ## pygame.surfarray.pixels3d ##

.. function:: array_alpha

   | :sl:`Copy pixel alphas into a 2d array`
   | :sg:`array_alpha(Surface) -> array`

   Copy the pixel alpha values (degree of transparency) from a Surface into a
   2D array. This will work for any type of Surface format. Surfaces without a
   pixel alpha will return an array with all opaque values.

   This function will temporarily lock the Surface as pixels are copied (see
   the :func:`Surface.lock` - lock the Surface memory for pixel access method).

   .. ## pygame.surfarray.array_alpha ##

.. function:: pixels_alpha

   | :sl:`Reference pixel alphas into a 2d array`
   | :sg:`pixels_alpha(Surface) -> array`

   Create a new 2D array that directly references the alpha values (degree of
   transparency) in a Surface. Any changes to the array will affect the pixels
   in the Surface. This is a fast operation since no data is copied.

   This can only work on 32-bit Surfaces with a per-pixel alpha value.

   The Surface this array references will remain locked for the lifetime of the
   array.

   .. ## pygame.surfarray.pixels_alpha ##

.. function:: pixels_red

   | :sl:`Reference pixel red into a 2d array.`
   | :sg:`pixels_red (Surface) -> array`

   Create a new 2D array that directly references the red values in a Surface.
   Any changes to the array will affect the pixels in the Surface. This is a
   fast operation since no data is copied.

   This can only work on 24-bit or 32-bit Surfaces.

   The Surface this array references will remain locked for the lifetime of the
   array.

   .. ## pygame.surfarray.pixels_red ##

.. function:: pixels_green

   | :sl:`Reference pixel green into a 2d array.`
   | :sg:`pixels_green (Surface) -> array`

   Create a new 2D array that directly references the green values in a
   Surface. Any changes to the array will affect the pixels in the Surface.
   This is a fast operation since no data is copied.

   This can only work on 24-bit or 32-bit Surfaces.

   The Surface this array references will remain locked for the lifetime of the
   array.

   .. ## pygame.surfarray.pixels_green ##

.. function:: pixels_blue

   | :sl:`Reference pixel blue into a 2d array.`
   | :sg:`pixels_blue (Surface) -> array`

   Create a new 2D array that directly references the blue values in a Surface.
   Any changes to the array will affect the pixels in the Surface. This is a
   fast operation since no data is copied.

   This can only work on 24-bit or 32-bit Surfaces.

   The Surface this array references will remain locked for the lifetime of the
   array.

   .. ## pygame.surfarray.pixels_blue ##

.. function:: array_colorkey

   | :sl:`Copy the colorkey values into a 2d array`
   | :sg:`array_colorkey(Surface) -> array`

   Create a new array with the colorkey transparency value from each pixel. If
   the pixel matches the colorkey it will be fully tranparent; otherwise it
   will be fully opaque.

   This will work on any type of Surface format. If the image has no colorkey a
   solid opaque array will be returned.

   This function will temporarily lock the Surface as pixels are copied.

   .. ## pygame.surfarray.array_colorkey ##

.. function:: make_surface

   | :sl:`Copy an array to a new surface`
   | :sg:`make_surface(array) -> Surface`

   Create a new Surface that best resembles the data and format on the array.
   The array can be 2D or 3D with any sized integer values. Function
   make_surface uses the array struct interface to aquire array properties,
   so is not limited to just NumPy arrays. See :mod:`pygame.pixelcopy`.

   New in Pygame 1.9.2: array struct interface support.

   .. ## pygame.surfarray.make_surface ##

.. function:: blit_array

   | :sl:`Blit directly from a array values`
   | :sg:`blit_array(Surface, array) -> None`

   Directly copy values from an array into a Surface. This is faster than
   converting the array into a Surface and blitting. The array must be the same
   dimensions as the Surface and will completely replace all pixel values. Only
   integer, ascii character and record arrays are accepted.

   This function will temporarily lock the Surface as the new values are
   copied.

   .. ## pygame.surfarray.blit_array ##

.. function:: map_array

   | :sl:`Map a 3d array into a 2d array`
   | :sg:`map_array(Surface, array3d) -> array2d`

   Convert a 3D array into a 2D array. This will use the given Surface format
   to control the conversion. Palette surface formats are supported for numpy
   arrays.

   .. ## pygame.surfarray.map_array ##

.. function:: use_arraytype

   | :sl:`Sets the array system to be used for surface arrays`
   | :sg:`use_arraytype (arraytype) -> None`

   Uses the requested array type for the module functions. Currently supported
   array types are:

   ::

     numeric (deprecated; will be removed in Pygame 1.9.3.)
     numpy

   If the requested type is not available, a ValueError will be raised.

   New in pygame 1.8.

   .. ## pygame.surfarray.use_arraytype ##

.. function:: get_arraytype

   | :sl:`Gets the currently active array type.`
   | :sg:`get_arraytype () -> str`

   Returns the currently active array type. This will be a value of the
   ``get_arraytypes()`` tuple and indicates which type of array module is used
   for the array creation.

   New in pygame 1.8

   .. ## pygame.surfarray.get_arraytype ##

.. function:: get_arraytypes

   | :sl:`Gets the array system types currently supported.`
   | :sg:`get_arraytypes () -> tuple`

   Checks, which array systems are available and returns them as a tuple of
   strings. The values of the tuple can be used directly in the
   :func:`pygame.surfarray.use_arraytype` () method. If no supported array
   system could be found, None will be returned.

   New in pygame 1.8.

   .. ## pygame.surfarray.get_arraytypes ##

.. ## pygame.surfarray ##
