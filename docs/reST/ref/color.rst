.. include:: common.txt

:mod:`pygame.Color`
===================

.. currentmodule:: pygame

.. class:: Color

   | :sl:`pygame object for color representations`
   | :sg:`Color(name) -> Color`
   | :sg:`Color(r, g, b, a) -> Color`
   | :sg:`Color(rgbvalue) -> Color`

   The Color class represents ``RGBA`` color values using a value range of
   0-255. It allows basic arithmetic operations — binary operations ``+``,
   ``-``, ``*``, ``//``, ``%``, and unary operation ``~`` — to create
   new colors, supports conversions to other color spaces such as ``HSV``
   or ``HSL`` and lets you adjust single color channels.
   Alpha defaults to 255 when not given.
   The arithmetic operations and ``correct_gamma()`` method preserve subclasses.
   For the binary operators, the class of the returned color is that of the
   left hand color object of the operator.

   'rgbvalue' can be either a color name, an ``HTML`` color format string, a
   hex number string, or an integer pixel value. The ``HTML`` format is
   '#rrggbbaa', where rr, gg, bb, and aa are 2 digit hex numbers. The alpha aa
   is optional. A hex number string has the form '0xrrggbbaa', where aa is
   optional.

   Color objects support equality comparison with other color objects and 3 or
   4 element tuples of integers (New in 1.9.0). There was a bug in pygame 1.8.1
   where the default alpha was 0, not 255 like previously.

   Color objects export the C level array interface. The interface exports a
   read-only one dimensional unsigned byte array of the same assigned length
   as the color. For CPython 2.6 and later, the new buffer interface is also
   exported, with the same characteristics as the array interface. New in
   pygame 1.9.2.

   The floor division, ``//``, and modulus, ``%``, operators do not raise
   an exception for division by zero. Instead, if a color, or alpha, channel
   in the right hand color is 0, then the result is 0. For example: ::

       # These expressions are True
       Color(255, 255, 255, 255) // Color(0, 64, 64, 64) == Color(0, 3, 3, 3)
       Color(255, 255, 255, 255) % Color(64, 64, 64, 0) == Color(63, 63, 63, 0)

   New implementation of Color was done in pygame 1.8.1.

   .. attribute:: r

      | :sl:`Gets or sets the red value of the Color.`
      | :sg:`r -> int`

      The red value of the Color.

      .. ## Color.r ##

   .. attribute:: g

      | :sl:`Gets or sets the green value of the Color.`
      | :sg:`g -> int`

      The green value of the Color.

      .. ## Color.g ##

   .. attribute:: b

      | :sl:`Gets or sets the blue value of the Color.`
      | :sg:`b -> int`

      The blue value of the Color.

      .. ## Color.b ##

   .. attribute:: a

      | :sl:`Gets or sets the alpha value of the Color.`
      | :sg:`a -> int`

      The alpha value of the Color.

      .. ## Color.a ##

   .. attribute:: cmy

      | :sl:`Gets or sets the CMY representation of the Color.`
      | :sg:`cmy -> tuple`

      The ``CMY`` representation of the Color. The ``CMY`` components are in
      the ranges ``C`` = [0, 1], ``M`` = [0, 1], ``Y`` = [0, 1]. Note that this
      will not return the absolutely exact ``CMY`` values for the set ``RGB``
      values in all cases. Due to the ``RGB`` mapping from 0-255 and the
      ``CMY`` mapping from 0-1 rounding errors may cause the ``CMY`` values to
      differ slightly from what you might expect.

      .. ## Color.cmy ##

   .. attribute:: hsva

      | :sl:`Gets or sets the HSVA representation of the Color.`
      | :sg:`hsva -> tuple`

      The ``HSVA`` representation of the Color. The ``HSVA`` components are in
      the ranges ``H`` = [0, 360], ``S`` = [0, 100], ``V`` = [0, 100], A = [0,
      100]. Note that this will not return the absolutely exact ``HSV`` values
      for the set ``RGB`` values in all cases. Due to the ``RGB`` mapping from
      0-255 and the ``HSV`` mapping from 0-100 and 0-360 rounding errors may
      cause the ``HSV`` values to differ slightly from what you might expect.

      .. ## Color.hsva ##

   .. attribute:: hsla

      | :sl:`Gets or sets the HSLA representation of the Color.`
      | :sg:`hsla -> tuple`

      The ``HSLA`` representation of the Color. The ``HSLA`` components are in
      the ranges ``H`` = [0, 360], ``S`` = [0, 100], ``V`` = [0, 100], A = [0,
      100]. Note that this will not return the absolutely exact ``HSL`` values
      for the set ``RGB`` values in all cases. Due to the ``RGB`` mapping from
      0-255 and the ``HSL`` mapping from 0-100 and 0-360 rounding errors may
      cause the ``HSL`` values to differ slightly from what you might expect.

      .. ## Color.hsla ##

   .. attribute:: i1i2i3

      | :sl:`Gets or sets the I1I2I3 representation of the Color.`
      | :sg:`i1i2i3 -> tuple`

      The ``I1I2I3`` representation of the Color. The ``I1I2I3`` components are
      in the ranges ``I1`` = [0, 1], ``I2`` = [-0.5, 0.5], ``I3`` = [-0.5,
      0.5]. Note that this will not return the absolutely exact ``I1I2I3``
      values for the set ``RGB`` values in all cases. Due to the ``RGB``
      mapping from 0-255 and the ``I1I2I3`` mapping from 0-1 rounding errors
      may cause the ``I1I2I3`` values to differ slightly from what you might
      expect.

      .. ## Color.i1i2i3 ##

   .. method:: normalize

      | :sl:`Returns the normalized RGBA values of the Color.`
      | :sg:`normalize() -> tuple`

      Returns the normalized ``RGBA`` values of the Color as floating point
      values.

      .. ## Color.normalize ##

   .. method:: correct_gamma

      | :sl:`Applies a certain gamma value to the Color.`
      | :sg:`correct_gamma (gamma) -> Color`

      Applies a certain gamma value to the Color and returns a new Color with
      the adjusted ``RGBA`` values.

      .. ## Color.correct_gamma ##

   .. method:: set_length

      | :sl:`Set the number of elements in the Color to 1,2,3, or 4.`
      | :sg:`set_length(len) -> None`

      The default Color length is 4. Colors can have lengths 1,2,3 or 4. This
      is useful if you want to unpack to r,g,b and not r,g,b,a. If you want to
      get the length of a Color do ``len(acolor)``.

      New in pygame 1.9.0.

      .. ## Color.set_length ##

   .. ## pygame.Color ##
