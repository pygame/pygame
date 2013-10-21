.. include:: common.txt

:mod:`pygame.freetype`
======================

.. module:: pygame.freetype
   :synopsis: Enhanced Pygame module for loading and rendering computer fonts

| :sl:`Enhanced Pygame module for loading and rendering computer fonts`

The :mod:`pygame.freetype` module allows for the rendering of all font file formats
supported by FreeType, namely ``TTF``, Type1, ``CFF``, OpenType, ``SFNT``, ``PCF``,
``FNT``, ``BDF``, ``PFR`` and Type42 fonts. It can render any UTF-32 character in a
font file.

This module is a replacement for :mod:`pygame.font`. It has all of the functionality
of the original, plus many new features. Yet is has absolutely no dependencies
on the SDL_ttf library. The :mod:`pygame.freetype` module is not itself backward
compatible with :mod:`pygame.font`. Instead, a new :mod:`pygame.ftfont` provides
a drop-in replacement for :mod:`pygame.font`.

Most of the work done with fonts is done by using the actual Font objects. The
module by itself only has routines to initialize itself and create Font objects
with ``pygame.freetype.Font()``.

You can load fonts from the system by using the ``pygame.freetype.SysFont()``
function. There are a few other functions to help find system fonts.

For now undefined character codes are replaced with the ``undefined character``.
How undefined codes are handled may become configurable in a future release.

Pygame comes with a builtin default font. This can always be accessed by
passing None as the font name to the Font constructor.

New in Pygame 1.9.2

.. function:: get_error

   | :sl:`Return the latest FreeType2 error`
   | :sg:`get_error() -> str`

   Return a description of the last error which occurred in the FreeType2
   library, or None if no errors have occurred.

.. function:: get_version

   | :sl:`Return the FreeType 2 version`
   | :sg:`get_version() -> (int, int, int)`

   Returns the version of the FreeType2 library which was used to build the
   'freetype' module.

   Note that the freetype module depends on the FreeType 2 library. It will
   not compile with the original FreeType 1.0. Hence, the first element of the
   tuple will always be "2".

.. function:: init

   | :sl:`Initialize the underlying FreeType 2 library.`
   | :sg:`init(cache_size=64, resolution=72)`

   This function initializes the underlying FreeType 2 library and must be
   called before trying to use any of the functionality of the 'freetype'
   module.

   However, this function will be automatically called by ``pygame.init()``.
   It is safe to call this function more than once.

   Optionally, you may specify a default size for the Glyph cache: this is the
   maximum number of glyphs that will be cached at any given time by the
   module. Exceedingly small values will be automatically tuned for
   performance. Also a default pixel resolution, in dots per inch, can
   be given to adjust font scaling.

.. function:: quit

   | :sl:`Shut down the underlying FreeType 2 library.`
   | :sg:`quit()`

   This function de-initializes the ``freetype`` module. After calling this
   function, you should not invoke any class, method or function related to the
   ``freetype`` module as they are likely to fail or might give unpredictable
   results. It is safe to call this function even if the module hasn't been
   initialized yet.

.. function:: was_init

   | :sl:`Return whether the the FreeType 2 library is initialized.`
   | :sg:`was_init() -> bool`

   Returns whether the the FreeType 2 library is initialized.

.. function:: get_default_resolution

   | :sl:`Return the default pixel size in dots per inch`
   | :sg:`get_default_resolution() -> long`

   Returns the default pixel size, in dots per inch for the module. If not changed
   it will be 72.

.. function:: set_default_resolution

   | :sl:`Set the default pixel size in dots per inch for the module`
   | :sg:`set_default_resolution([resolution])`

   Set the default pixel size, in dots per inch, for the module. If the
   optional argument is omitted or zero the resolution is reset to 72.

.. function:: get_default_font

   | :sl:`Get the filename of the default font`
   | :sg:`get_default_font() -> string`

   Return the filename of the system font. This is not the full path to the
   file. This file can usually be found in the same directory as the font
   module, but it can also be bundled in separate archives.

.. class:: Font

   | :sl:`Create a new Font instance from a supported font file.`
   | :sg:`Font(file, size=0, font_index=0, resolution=0, ucs4=False) -> Font`

   Argument *file* can be either a string representing the font's filename, a
   file-like object containing the font, or None; if None, the default, built-in font
   is used.

   Optionally, a *size* argument may be specified to set the default size in
   points, which will be used when rendering the font. The size can also be
   passed explicitly to each method call. Because of the way the caching
   system works, specifying a default size on the constructor doesn't imply a
   performance gain over manually passing the size on each function call.
   If the font is bitmap and no *size* is given, the default size is set
   to the first available size for the font, if possible.

   If the font file has more than one font, the font to load can be chosen with
   the *index* argument. An exception is raised for an out-of-range font index
   value.

   The optional resolution argument sets the pixel size, in dots per inch,
   for use in scaling glyphs for this Font instance. If 0 then the default
   module value, set by :meth:`freetype.init`, is used. The Font object's
   resolution can only be changed by reinitializing the Font instance.

   The optional ucs4 argument, an integer, sets the default text translation
   mode: 0 (False) recognize UTF-16 surrogate pairs, any other value (True),
   to treat Unicode text as UCS-4, with no surrogate pairs. See
   :attr:`Font.ucs4`.

   .. attribute:: name

      | :sl:`Proper font name.`
      | :sg:`name -> string`

      Read only. Returns the real (long) name of the font, as
      recorded in the font file.

   .. attribute:: path

      | :sl:`Font file path`
      | :sg:`path -> unicode`

      Read only. Returns the path of the loaded font file

   .. attribute:: size

      | :sl:`The default point size used in rendering`
      | :sg:`size -> float`
      | :sg:`size -> (float, float)`

      Get or set the default size for text metrics and rendering. It can be
      a single point size, given as an Python ``int`` or ``float``, or a
      font ppem (width, height) ``tuple``. Size values are non-negative.
      A zero size or width represents an undefined size. In this case
      the size must be given as a method argument, or an exception is
      raised. A zero width but non-zero height is a ValueError.

      For a scalable font, a single number value is equivalent to a tuple
      with width equal height. A font can be stretched vertically with
      height set greater than width, or horizontally with width set
      greater than height. For embedded bitmaps, as listed by :meth:`get_sizes`,
      use the nominal width and height to select an available size.

      Font size differs for a non-scalable, bitmap, font. During a
      method call it must match one of the available sizes returned by
      method :meth:`get_sizes`. If not, an exception is raised.
      If the size is a single number, the size is first matched against the
      point size value. If no match, then the available size with the
      same nominal width and height is chosen.

   .. method:: get_rect

      | :sl:`Return the size and offset of rendered text`
      | :sg:`get_rect(text, style=STYLE_DEFAULT, rotation=0, size=0) -> rect`

      Gets the final dimensions and origin, in pixels, of 'text' using the
      current point size, style, rotation and orientation. These are either
      taken from the arguments, if given, else from the default values set
      for the font object.

      Returns a rect containing the width and height of the text's bounding
      box and the position of the text's origin. The origin can be used
      to align separately rendered pieces of text. It gives the baseline
      position and bearing at the start of the text.

      If text is a char (byte) string, then its encoding is assumed to be
      ``LATIN1``.

   .. method:: get_metrics

      | :sl:`Return the glyph metrics for the given text`
      | :sg:`get_metrics(text, size=0) -> [(...), ...]`

      Returns the glyph metrics for each character in 'text'.

      The glyph metrics are returned inside a list; each character will be
      represented as a tuple inside the list with the following values:

      ::

          (min_x, max_x, min_y, max_y, horizontal_advance_x, horizontal_advance_y)

      The bounding box min_x, max_y, min_y, and max_y values are returned as
      grid-fitted pixel coordinates of type int. The advance values are 
      float values.

      The calculations are done using the font's default size in points.
      Optionally you may specify another point size to use.

      The metrics are adjusted for the current rotation, strong, and oblique
      settings.

      If text is a char (byte) string, then its encoding is assumed to be
      ``LATIN1``.

   .. attribute:: height

      | :sl:`The unscaled height of the font in font units`
      | :sg:`height -> int`

      Read only. Gets the height of the font. This is the average value of all
      glyphs in the font.

   .. attribute:: ascender

      | :sl:`The unscaled ascent of the font in font units`
      | :sg:`ascender -> int`

      Read only. Return the number of units from the font's baseline to
      the top of the bounding box.

   .. attribute:: descender

      | :sl:`The unscaled descent of the font in font units`
      | :sg:`descender -> int`

      Read only. Return the height in font units for the font descent.
      The descent is the number of units from the font's baseline to the
      bottom of the bounding box.

   .. method:: get_sized_ascender

      | :sl:`The scaled ascent of the font in pixels`
      | :sg:`get_sized_ascender(<size>=0) -> int`

      Return the number of units from the font's baseline to the top of the
      bounding box. It is not adjusted for strong or rotation.

   .. method:: get_sized_descender

      | :sl:`The scaled descent of the font in pixels`
      | :sg:`get_sized_descender(<size>=0) -> int`

      Return the number of pixels from the font's baseline to the top of the
      bounding box. It is not adjusted for strong or rotation.

   .. method:: get_sized_height

      | :sl:`The scaled height of the font in pixels`
      | :sg:`get_sized_height(<size>=0) -> int`

      Read only. Gets the height of the font. This is the average value of all
      glyphs in the font. It is not adjusted for strong or rotation.

   .. method:: get_sized_glyph_height

      | :sl:`The scaled bounding box height of the font in pixels`
      | :sg:`get_sized_glyph_height(<size>=0) -> int`

      Return the glyph bounding box height of the font in pixels.
      This is the average value of all glyphs in the font.
      It is not adjusted for strong or rotation.

   .. method:: get_sizes

      | :sl:`return the available sizes of embedded bitmaps`
      | :sg:`get_sizes() -> [(int, int, int, float, float), ...]`
      | :sg:`get_sizes() -> []`

      This returns a list of tuple records, one for each point size
      supported. Each tuple containing the point size, the height in pixels,
      width in pixels, horizontal ppem (nominal width) in fractional pixels,
      and vertical ppem (nominal height) in fractional pixels.

   .. method:: render

      | :sl:`Return rendered text as a surface`
      | :sg:`render(text, fgcolor, bgcolor=None, style=STYLE_DEFAULT, rotation=0, size=0) -> (Surface, Rect)`

      Returns a new :mod:`pygame.Surface`, with the text rendered to it
      in the color given by 'fgcolor'. If ``bgcolor`` is given, the surface
      will be filled with this color. If no background color is given,
      the surface is filled with zero alpha opacity. Normally the returned
      surface has a 32 bit pixel size. However, if ``bgcolor`` is ``None``
      and anti-aliasing is disabled a two color 8 bit surface with colorkey
      set for the background color is returned.

      The return value is a tuple: the new surface and the bounding
      rectangle giving the size and origin of the rendered text.

      If an empty string is passed for text then the returned Rect is zero
      width and the height of the font. If dest is None the returned surface is
      the same dimensions as the boundary rect. The rect will test False.

      The rendering is done using the font's default size in points and its
      default style, without any rotation, and taking into account fonts which
      are set to be drawn vertically via the :attr:`vertical` attribute.
      Optionally you may specify another point size to use via the 'size'
      argument, a text rotation via the 'rotation' argument, or a new text
      style via the 'style' argument. See the attr :attr:`size`,
      :attr:`rotation`, and :attr:`style` attributes.

      If text is a char (byte) string, then its encoding is assumed to be
      ``LATIN1``.

   .. method:: render_to

      | :sl:`Render text onto an existing surface`
      | :sg:`render(surf, dest, text, fgcolor, bgcolor=None, style=STYLE_DEFAULT, rotation=0, size=0) -> Rect`

      Renders the string 'text' to a :mod:`pygame.Surface` 'surf',
      using the color 'fgcolor'.

      Argument 'dest' is an (x, y) surface coordinate pair. If either x
      or y is not an integer it is converted to one if possible.
      Any sequence, including Rect, for which the first two elements are
      positions x and y is accepted.

      If a background color is given, the surface is first filled with that
      color. The text is blitted next. Both the background fill and text
      rendering involve full alpha blits. That is, the alpha values of
      both the foreground and background colors, as well as those of the
      destination surface if it has per-pixel alpha.

      The return value is a rectangle giving the size and position of the
      rendered text within the surface.

      If an empty string is passed for text then the returned Rect is zero
      width and the height of the font. The rect will test False.

      By default, the point size and style set for the font are used
      if not passed as arguments. The text is unrotated unless a non-zero
      rotation value is given.

      If text is a char (byte) string, then its encoding is assumed to be
      ``LATIN1``.

   .. method:: render_raw

      | :sl:`Return rendered text as a string of bytes`
      | :sg:`render_raw(text, style=STYLE_DEFAULT, rotation=0, size=0, invert=False) -> (bytes, (int, int))`

      Like ``Font.render()`` but the tuple returned is an 8 bit
      monochrome string of bytes and its size. The foreground color is 255, the
      background 0, useful as an alpha mask for a foreground pattern.

   .. method:: render_raw_to

      | :sl:`Render text into an array of ints`
      | :sg:`render_raw_to(array, text, dest=None, style=STYLE_DEFAULT, rotation=0, size=0, invert=False) -> (int, int)`

      Render to an array object exposing an array struct interface. The array
      must be two dimensional with integer items. The default dest value, None,
      is equivalent to (0, 0).

   .. attribute:: style

      | :sl:`The font's style flags`
      | :sg:`style -> int`

      Gets or sets the default style of the Font. This default style will be
      used for all text rendering and size calculations unless overridden
      specifically in the \`render()` or \`get_size()` calls. The style value
      may be a bit-wise ``OR`` of one or more of the following constants:

      ::

          STYLE_NONE
          STYLE_UNDERLINE
          STYLE_OBLIQUE
          STYLE_STRONG
	  STYLE_WIDE

      These constants may be found on the FreeType constants module.
      Optionally, the default style can be modified or obtained accessing the
      individual style attributes (underline, oblique, strong).

      The ``STYLE_OBLIQUE`` and ``STYLE_STRONG`` styles are for scalable fonts
      only. An attempt to set either for a bitmap font raises an AttributeError.
      An attempt to set either for an inactive font, as returned by
      ``Font.__new__()``, raises a RuntimeError.

   .. attribute:: underline

      | :sl:`The state of the font's underline style flag`
      | :sg:`underline -> bool`

      Gets or sets whether the font will be underlined when drawing text. This
      default style value will be used for all text rendering and size
      calculations unless overridden specifically in the \`render()` or
      \`get_size()` calls, via the 'style' parameter.

   .. attribute:: strong

      | :sl:`The state of the font's strong style flag`
      | :sg:`strong -> bool`

      Gets or sets whether the font will be bold when drawing text. This
      default style value will be used for all text rendering and size
      calculations unless overridden specifically in the \`render()` or
      \`get_size()` calls, via the 'style' parameter.

   .. attribute:: oblique

      | :sl:`The state of the font's oblique style flag`
      | :sg:`oblique -> bool`

      Gets or sets whether the font will be rendered as oblique. This
      default style value will be used for all text rendering and size
      calculations unless overridden specifically in the \`render()` or
      \`get_size()` calls, via the 'style' parameter.

      The oblique style is only supported for scalable (outline) fonts.
      An attempt to set this property will raise an AttributeError.
      If the font object is inactive, as returned by Font.__new__,
      setting this property raises a RuntimeError.

   .. attribute:: wide

      | :sl:`The state of the font's wide style flag`
      | :sg:`wide -> bool`

      Gets or sets whether the font will be stretched horizontally
      when drawing text. It produces a result similar to font.Font's
      bold. This style is only available for unrotated text.

   .. attribute:: strength

      | :sl:`The strength associated with the strong or wide font styles`
      | :sg:`strength -> float`

      The amount by which a font glyph's size is enlarged for the
      strong or wide transformations, as a fraction of the untransformed
      size. For the wide style only the horizontal dimension is
      increased. For strong text both the horizontal and vertical
      dimensions are enlarged. A wide style of strength 1/12 is
      equivalent to the font.Font bold style. The default is 1/36.

      The strength style is only supported for scalable (outline) fonts.
      An attempt to set this property will raise an AttributeError.
      If the font object is inactive, as returned by Font.__new__,
      setting this property raises a RuntimeError.

   .. attribute:: underline_adjustment

      | :sl:`Adjustment factor for the underline position`
      | :sg:`underline_adjustment -> float`

      Gets or sets a factor which, when positive, is multiplied with the
      font's underline offset to adjust the underline position. A negative
      value turns an underline into a strike-through or overline. It is
      multiplied with the ascender. Accepted values are between -2.0 and 2.0
      inclusive. A value of 0.5 closely matches Tango underlining. A value of
      1.0 mimics SDL_ttf.

   .. attribute:: fixed_width

      | :sl:`Gets whether the font is fixed-width`
      | :sg:`fixed_width -> bool`

      Read only. Return True if the font contains fixed-width characters
      (for example Courier, Bitstream Vera Sans Mono, Andale Mono).

   .. attribute:: fixed_sizes

      | :sl:`the number of embedded bitmap sizes the font`
      | :sg:`fixed_sizes -> int`

      Read only. Return the number of point sizes for which the font contains
      bitmap character images. If zero then the font is not a bitmap font. 
      A scalable font may contain pre-rendered point sizes.

   .. attribute:: scalable

      | :sl:`Gets whether the font is scalable`
      | :sg:`scalable -> bool`

      Read only. Return True if the font contains outline glyphs. If so,
      the point size is not limited to available bitmap sizes.

   .. attribute:: use_bitmap_strikes

      | :sl:`allow the use of embeddeded bitmaps in an outline font file`
      | :sg:`use_bitmap_strikes -> bool`

      Some scalable fonts contain embedded bitmaps for particular point
      sizes. This property controls whether or not those bitmap strikes
      are used. Setting ``False`` disables the loading of any bitmap strike.
      Setting ``True``, the default value, allows bitmap strikes for an
      unrotated render when no style other than :attr:`wide` or
      :attr:`underline` is set. This property has no effect on bitmap fonts.

      See also :attr:`fixed_sizes` and :meth:`get_sizes`.

   .. attribute:: antialiased

      | :sl:`Font anti-aliasing mode`
      | :sg:`antialiased -> bool`

      Gets or sets the font's anti-aliasing mode. This defaults to ``True`` on
      all fonts, which are rendered with full 8 bit blending.

      Setting this to ``False`` will enable monochrome rendering. This should
      provide a small speed gain and reduce cache memory size.

   .. attribute:: kerning

      | :sl:`Character kerning mode`
      | :sg:`kerning -> bool`

      Gets or sets the font's kerning mode. This defaults to False on all
      fonts, which will be rendered by default without kerning.

      Setting this to true will change all rendering methods to do kerning
      between character pairs for surface size calculation and all
      render operations.

   .. attribute:: vertical

      | :sl:`Font vertical mode`
      | :sg:`vertical -> bool`

      Gets or sets whether the font is a vertical font such as fonts in fonts
      representing Kanji glyphs or other styles of vertical writing.

      Changing this attribute will cause the font to be rendering vertically,
      and affects all other methods which manage glyphs or text layouts to use
      vertical metrics accordingly.

      Note that the FreeType library doesn't automatically detect whether a
      font contains glyphs which are always supposed to be drawn vertically, so
      this attribute must be set manually by the user.

      Also note that several font formats (especially bitmap based ones) don't
      contain the necessary metrics to draw glyphs vertically, so drawing in
      those cases will give unspecified results.

   .. attribute:: rotation

      | :sl:`text rotation in degrees counterclockwise`
      | :sg:`rotation -> int`

      Get or set the baseline angle of the rendered text. The angle is
      represented as integer degrees. The default angle is 0, with horizontal
      text rendered along the X axis, and vertical text along the Y axis.
      A non-zero value rotates these axes counterclockwise that many degrees.
      Degree values outside of the range 0 to 359 inclusive are reduced to the
      corresponding angle within the range (eg. 390 -> 390 - 360 -> 30,
      -45 -> 360 + -45 -> 315, 720 -> 720 - (2 * 360) -> 0).

      Text rotation is only supported for scalable (outline) fonts. An attempt
      to change the rotation of a bitmap font raises an AttributeError.
      An attempt to change the rotation of an inactive font objects, as
      returned by Font.__new__(), raises a RuntimeError.

   .. attribute:: origin

      | :sl:`Font render to text origin mode`
      | :sg:`origin -> bool`

      If set True, then when rendering to an existing surface, the position
      is taken to be that of the text origin. Otherwise the render position is
      the top-left corner of the text bounding box.

   .. attribute:: pad

      | :sl:`padded boundary mode`
      | :sg:`pad -> bool`

      If set True, then the text boundary rectangle will be inflated to match
      that of font.Font. Otherwise, the boundary rectangle is just large
      enough for the text.

   .. attribute:: ucs4

      | :sl:`Enable UCS-4 mode`
      | :sg:`ucs4 -> bool`

      Gets or sets the decoding of Unicode text. By default, the
      freetype module performs UTF-16 surrogate pair decoding on Unicode text.
      This allows 32-bit escape sequences ('\Uxxxxxxxx') between 0x10000 and
      0x10FFFF to represent their corresponding UTF-32 code points on Python
      interpreters built with a UCS-2 unicode type (on Windows, for instance).
      It also means character values within the UTF-16 surrogate area (0xD800
      to 0xDFFF) are considered part of a surrogate pair. A malformed surrogate
      pair will raise an UnicodeEncodeError. Setting ucs4 True turns surrogate
      pair decoding off, letting interpreters with a UCS-4 unicode type access
      the full UCS-4 character range.

   .. attribute:: resolution

      | :sl:`Pixel resolution in dots per inch`
      | :sg:`resolution -> int`

      Gets the pixel size used in scaling font glyphs for this Font instance.
