.. include:: common.txt

:mod:`pygame.Color`
===================

.. currentmodule:: pygame

.. class:: Color

   | :sl:`pygame object for color representations`
   | :sg:`Color(r, g, b) -> Color`
   | :sg:`Color(r, g, b, a=255) -> Color`
   | :sg:`Color(color_value) -> Color`

   The ``Color`` class represents ``RGBA`` color values using a value range of
   0 to 255 inclusive. It allows basic arithmetic operations — binary
   operations ``+``, ``-``, ``*``, ``//``, ``%``, and unary operation ``~`` — to
   create new colors, supports conversions to other color spaces such as ``HSV``
   or ``HSL`` and lets you adjust single color channels.
   Alpha defaults to 255 (fully opaque) when not given.
   The arithmetic operations and ``correct_gamma()`` method preserve subclasses.
   For the binary operators, the class of the returned color is that of the
   left hand color object of the operator.

   Color objects support equality comparison with other color objects and 3 or
   4 element tuples of integers. There was a bug in pygame 1.8.1
   where the default alpha was 0, not 255 like previously.

   Color objects export the C level array interface. The interface exports a
   read-only one dimensional unsigned byte array of the same assigned length
   as the color. For CPython 2.6 and later, the new buffer interface is also
   exported, with the same characteristics as the array interface.

   The floor division, ``//``, and modulus, ``%``, operators do not raise
   an exception for division by zero. Instead, if a color, or alpha, channel
   in the right hand color is 0, then the result is 0. For example: ::

       # These expressions are True
       Color(255, 255, 255, 255) // Color(0, 64, 64, 64) == Color(0, 3, 3, 3)
       Color(255, 255, 255, 255) % Color(64, 64, 64, 0) == Color(63, 63, 63, 0)

   :param int r: red value in the range of 0 to 255 inclusive
   :param int g: green value in the range of 0 to 255 inclusive
   :param int b: blue value in the range of 0 to 255 inclusive
   :param int a: (optional) alpha value in the range of 0 to 255 inclusive,
      default is 255
   :param color_value: color value (see note below for the supported formats)

      .. note::
         Supported ``color_value`` formats:
            | - **Color object:** clones the given :class:`Color` object
            | - **color name str:** name of the color to use, e.g. ``'red'``
              (all the supported name strings can be found in the
              `colordict module <https://github.com/pygame/pygame/blob/master/src_py/colordict.py>`_)
            | - **HTML color format str:** ``'#rrggbbaa'`` or ``'#rrggbb'``,
              where rr, gg, bb, and aa are 2-digit hex numbers in the range
              of 0 to 0xFF inclusive, the aa (alpha) value defaults to 0xFF
              if not provided
            | - **hex number str:** ``'0xrrggbbaa'`` or ``'0xrrggbb'``, where
              rr, gg, bb, and aa are 2-digit hex numbers in the range of 0x00
              to 0xFF inclusive, the aa (alpha) value defaults to 0xFF if not
              provided
            | - **int:** int value of the color to use, using hex numbers can
              make this parameter more readable, e.g. ``0xrrggbbaa``, where rr,
              gg, bb, and aa are 2-digit hex numbers in the range of 0x00 to
              0xFF inclusive, note that the aa (alpha) value is not optional for
              the int format and must be provided
            | - **tuple/list of int color values:** ``(R, G, B, A)`` or
              ``(R, G, B)``, where R, G, B, and A are int values in the range of
              0 to 255 inclusive, the A (alpha) value defaults to 255 if not
              provided

   :type color_value: Color or str or int or tuple(int, int, int, [int]) or
      list(int, int, int, [int])

   :returns: a newly created :class:`Color` object
   :rtype: Color

   .. versionchanged:: 2.0.0
      Support for tuples, lists, and :class:`Color` objects when creating
      :class:`Color` objects.
   .. versionchanged:: 1.9.2 Color objects export the C level array interface.
   .. versionchanged:: 1.9.0 Color objects support 4-element tuples of integers.
   .. versionchanged:: 1.8.1 New implementation of the class.

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

      .. versionadded:: 1.9.0

      .. ## Color.set_length ##

   .. method:: lerp

      | :sl:`returns a linear interpolation to the given Color.`
      | :sg:`lerp(Color, float) -> Color`

      Returns a Color which is a linear interpolation between self and the
      given Color in RGBA space. The second parameter determines how far
      between self and other the result is going to be.
      It must be a value between 0 and 1 where 0 means self and 1 means
      other will be returned.

      .. versionadded:: 2.0.1

      .. ## Color.lerp ##

   .. ## pygame.Color ##
