.. include:: common.txt

:mod:`pygame.freetype`
======================

.. module:: pygame.freetype
   :synopsis: Enhanced Pygame module for loading and rendering font faces

| :sl:`Enhanced Pygame module for loading and rendering font faces`

--- Note that some features may change before a formal release

This module allows for rendering all face formats supported by FreeType, namely
``TTF``, Type1, ``CFF``, OpenType, ``SFNT``, ``PCF``, ``FNT``, ``BDF``, ``PFR``
and Type42 faces. It can render any UTF-32 character in a font file.

This module is optional, and replaces all of the functionality of the original
'font' module, whilst expanding it. This module depends in no way on the
SDL_ttf library.

You should test that :mod:`pygame.freetype` is initialized before attempting to
use the module; if the module is available and loaded, it will be automatically
initialized by ``pygame.init()``

Most of the work done with faces is done by using the actual Face objects. The
module by itself only has routines to initialize itself and create Face objects
with ``pygame.freetype.Face()``.

You can load faces from the system by using the ``pygame.freetype.SysFont()``
function. There are a few other functions to help lookup the system fonts.

For now undefined character codes are replaced with the ``undefined character``.
How undefined codes are handled may become configurable in a future release.

Pygame comes with a builtin default font. This can always be accessed by
passing None as the font name to the Face constructor.

New in Pygame 1.9.2

.. function:: get_error

   | :sl:`Get the latest error`
   | :sg:`get_error() -> str`

   Returns the description of the last error which occurred in the FreeType
   library, or None if no errors have occurred.

.. function:: get_version

   | :sl:`Get the FreeType version`
   | :sg:`get_version() -> (int, int, int)`

   Gets the version of the FreeType2 library which was used to build the
   'freetype' module.

   Note that the FreeType module depends on the FreeType 2 library, and will
   not compile with the original FreeType 1.0, hence the first element of the
   tuple will always be 2.

.. function:: init

   | :sl:`Initialize the underlying FreeType 2 library.`
   | :sg:`init(cache_size=64, resolution=72) -> None`

   This function initializes the underlying FreeType 2 library and must be
   called before trying to use any of the functionality of the 'freetype'
   module.

   However, if the module is available, this function will be automatically
   called by ``pygame.init()``. It is safe to call this function more than
   once.

   Optionally, you may specify a default size for the Glyph cache: this is the
   maximum amount of glyphs that will be cached at any given time by the
   module. Exceedingly small values will be automatically tuned for
   performance. Also a default pixel resolution, in dots per inch, can
   be given to adjust face scaling.

.. function:: quit

   | :sl:`Shuts down the underlying FreeType 2 library.`
   | :sg:`quit() -> None`

   This function de-initializes the 'freetype' module. After calling this
   function, you should not invoke any class, method or function related to the
   'freetype' module as they are likely to fail or might give unpredictable
   results. It is safe to call this function even if the module hasn't been
   initialized yet.

.. function:: was_init

   | :sl:`Returns whether the the FreeType 2 library is initialized.`
   | :sg:`was_init() -> bool`

   Returns whether the the FreeType 2 library is initialized.

.. function:: get_default_resolution

   | :sl:`Return the default pixel size in dots per inch`
   | :sg:`get_default_resolution() -> long`

   Returns the default pixel size, in dots per inch for the module. At
   initial module load time the value is 72.

.. function:: set_default_resolution

   | :sl:`Set the default pixel size in dots per inch for the module`
   | :sg:`set_default_resolution([resolution]) -> None`

   Set the default pixel size, in dots per inch, for the module. If the
   optional argument is omitted or zero the resolution is reset to 72.

.. function:: get_default_font

   | :sl:`Get the filename of the default font`
   | :sg:`get_default_font() -> string`

   Return the filename of the system font. This is not the full path to the
   file. This file can usually be found in the same directory as the font
   module, but it can also be bundled in separate archives.

.. class:: Face

   | :sl:`Creates a new Face instance from a supported font file.`
   | :sg:`Face(file, style=STYLE_NONE, ptsize=-1, face_index=0, vertical=0, ucs4=0, resolution=0) -> Face`

   'file' can be either a string representing the font's filename, a file-like
   object containing the font, or None; in this last case the default, built-in
   font will be used.

   Optionally, a \*ptsize* argument may be specified to set the default size in
   points which will be used to render the face. Such size can also be
   specified manually on each method call. Because of the way the caching
   system works, specifying a default size on the constructor doesn't imply a
   performance gain over manually passing the size on each function call.

   If the font file has more than one face, the \*index* argument may be
   specified to specify which face index to load. Defaults to 0; face loading
   will fail if the given index is not contained in the font.

   The 'style' argument will set the default style (oblique, underline, strong)
   used to draw this face. This style may be overriden on any ``Face.render()``
   call.

   The optional vertical argument, an integer, sets the default orientation
   for the face: 0 (False) for horizontal, any other value (True) for vertical.
   See :attr:`Face.vertical`.

   The optional ucs4 argument, an integer, sets the default text translation
   mode: 0 (False) recognize UTF-16 surrogate pairs, any other value (True),
   to treat unicode text as UCS-4, with no surrogate pairs. See
   :attr:`Face.ucs4`.

   The optional resolution argument sets the pixel size, in dots per inch,
   to use for scaling glyphs for this Face instance. If 0 then the default
   module value, set by :meth:`freetype.init`, is used. The Face object's
   resolution can only be changed by reinitializing the instance.

   .. attribute:: name

      | :sl:`Gets the name of the font face.`
      | :sg:`name -> string`

      Read only. Returns the real (long) name of the font type face, as
      specified on the font file.

   .. attribute:: path

      | :sl:`Gets the path of the font file`
      | :sg:`path -> unicode`

      Read only. Returns the path of the loaded font file

   .. method:: get_rect

      | :sl:`Gets the size and offset of rendered text`
      | :sg:`get_rect(text, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> rect`

      Gets the final dimensions and origin, in pixels, of 'text' using the
      current point size, style, rotation and orientation. These are either
      taken from the arguments, if given, else from the default values set
      for the face object.

      Returns a rect containing the width and height of the text's bounding
      box and the position of the text's origin. The origin can be used
      to align separately rendered pieces of text. It gives the baseline
      position and bearing at the start of the text.

      If text is a char (byte) string, then its encoding is assumed to be
      ``LATIN1``.

   .. method:: get_metrics

      | :sl:`Gets glyph metrics for the face's characters`
      | :sg:`get_metrics(text, ptsize=default) -> [(...), ...]`

      Returns the glyph metrics for each character in 'text'.

      The glyph metrics are returned inside a list; each character will be
      represented as a tuple inside the list with the following values:

      ::

          (min_x, max_x, min_y, max_y, horizontal_advance_x, horizontal_advance_y)

      The bounding box min_x, max_y, min_y, and max_y values are returned as
      grid-fitted pixel coordinates of type int. The advance values are 
      float values.

      The calculations are done using the face's default size in points.
      Optionally you may specify another point size to use.

      The metrics are adjusted for the current rotation, strong, and oblique
      settings.

      If text is a char (byte) string, then its encoding is assumed to be
      ``LATIN1``.

   .. attribute:: height

      | :sl:`Gets the unscaled height of the face in font units`
      | :sg:`height -> int`

      Read only. Gets the height of the face. This is the average value of all
      glyphs in the face.

   .. method:: ascender

      | :sl:`get the unscaled ascent of the face in font units`
      | :sg:`ascender -> int`

      Read only. Return the number of units from the face's baseline to
      the top of the bounding box.

   .. attribute:: descender

      | :sl:`get the unscaled descent of the face in font units`
      | :sg:`descender -> int`

      Read only. Return the height in font units for the face descent.
      The descent is the number of units from the face's baseline to the
      bottom of the bounding box.

   .. attribute:: get_sized_ascender

      | :sl:`Gets the scaled ascent the face in pixels`
      | :sg:`get_sized_ascender() -> int`

      Return the number of units from the face's baseline to the top of the
      bounding box. It is not adjusted for strong or rotation.

   .. method:: get_sized_descender

      | :sl:`Gets the scaled descent the face in pixels`
      | :sg:`get_sized_descender() -> int`

      Return the number of pixels from the face's baseline to the top of the
      bounding box. It is not adjusted for strong or rotation.

   .. attribute:: get_sized_height

      | :sl:`Gets the scaled height of the face in pixels`
      | :sg:`get_sized_height() -> int`

      Read only. Gets the height of the face. This is the average value of all
      glyphs in the face. It is not adjusted for strong or rotation.

   .. method:: get_sized_glyph_height

      | :sl:`Gets the scaled height of the face in pixels`
      | :sg:`get_sized_glyph_height() -> int`

      Return the glyph bounding box height of the face in pixels.
      This is the average value of all glyphs in the face.
      It is not adjusted for strong or rotation.

   .. method:: render

      | :sl:`Renders text on a surface`
      | :sg:`render(dest, text, fgcolor, bgcolor=None, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> (Surface, Rect)`

      Renders the string 'text' to a :mod:`pygame.Surface`, using the color
      'fgcolor'.

      The 'dest' parameter is supposed to be a sequence containing the surface
      and the coordinates at which the text will be rendered, in that order.
      The sequence may be either (surf, posn) or (surf, x, y), where x and y
      are numbers. posn can be any sequence, including Rect, for which the
      first two elements are positions x and y. If x and y are not integers
      they will be cast to int: ``int(x)``, ``int(y)``.

      If such a sequence exists, and the destination surface is a valid
      :mod:`pygame.Surface` (independently of its bit depth), the text will be
      rendered directly on top of it at the passed coordinates, using the given
      'fgcolor', and painting the background of the text with the given
      'bgcolor', if available. The alpha values for both colors are always
      taken into account.

      If 'None' is passed instead of a destination sequence, a new 
      :mod:`pygame.Surface` will be created with the required size to contain
      the drawn text, and using ``bgcolor`` as its background color. If a
      background color is not available, the surface will be filled with zero
      alpha opacity. Normally the returned surface has a 32 bit pixel size.
      However, if ``bgcolor`` is ``None`` and antialiasing is disabled
      a two color 8 bit surface with colorkey set for the background color
      is returned.

      The return value is a tuple: the target surface and the bounding
      rectangle giving the size and position of the rendered text within the
      surface.

      If an empty string is passed for text then the returned Rect is zero
      width and the height of the face. If dest is None the returned surface is
      the same dimensions as the boundary rect. The rect will test False.

      The rendering is done using the face's default size in points and its
      default style, without any rotation, and taking into account faces which
      are set to be drawn vertically via the :meth:`Face.vertical` attribute.
      Optionally you may specify another point size to use via the 'ptsize'
      argument, a text rotation via the 'rotation' argument, or a new text
      style via the 'style' argument.

      If text is a char (byte) string, then its encoding is assumed to be
      ``LATIN1``.

   .. method:: render_raw

      | :sl:`Renders text as a string of bytes`
      | :sg:`render_raw(text, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> (bytes, (int, int))`

      Like ``Face.render(None, ...)`` but the tuple returned is an 8 bit
      monochrome string of bytes and its size. The forground color is 255, the
      background 0, useful as an alpha mask for a foreground pattern.

   .. attribute:: style

      | :sl:`Gets or sets the face's style`
      | :sg:`style -> int`

      Gets or sets the default style of the Face. This default style will be
      used for all text rendering and size calculations unless overriden
      specifically in the \`render()` or \`get_size()` calls. The style value
      may be a bitwise ``OR`` of one or more of the following constants:

      ::

          STYLE_NONE
          STYLE_UNDERLINE
          STYLE_OBLIQUE
          STYLE_STRONG
	  STYLE_WIDE

      These constants may be found on the FreeType constants module.
      Optionally, the default style can be modified or obtained accessing the
      individual style attributes (underline, oblique, strong).

   .. attribute:: underline

      | :sl:`Gets or sets the face's underline style`
      | :sg:`underline -> bool`

      Gets or sets whether the face will be underlined when drawing text. This
      default style value will be used for all text rendering and size
      calculations unless overriden specifically in the \`render()` or
      \`get_size()` calls, via the 'style' parameter.

   .. attribute:: strong

      | :sl:`Gets or sets the face's strong style`
      | :sg:`strong -> bool`

      Gets or sets whether the face will be bold when drawing text. This
      default style value will be used for all text rendering and size
      calculations unless overriden specifically in the \`render()` or
      \`get_size()` calls, via the 'style' parameter.

   .. attribute:: oblique

      | :sl:`Gets or sets the face's oblique style`
      | :sg:`oblique -> bool`

      Gets or sets whether the face will be rendered as oblique. This
      default style value will be used for all text rendering and size
      calculations unless overriden specifically in the \`render()` or
      \`get_size()` calls, via the 'style' parameter.

   .. attribute:: wide

      | :sl:`Gets or sets the face's wide style`
      | :sg:`wide -> bool`

      Gets or sets whether the face will be stretched horizontally
      when drawing text. It produces a result simular to font.Font's
      bold. This style is only available for unrotated text.

   .. attribute:: strength

      | :sl:`Gets or sets the strength of the strong or wide styles`
      | :sg:`strength -> float`

      The amount by which a face glyph's size is enlarged for the
      strong or wide transformations, as a fraction of the untransformed
      size. For the wide style only the horizontal dimension is
      increased. For strong text both the horizontal and vertical
      dimensions are enlarged. A wide style of strength 1/12 is
      equivalent to the font.Font bold style. The default is 1/36.

   .. attribute:: underline_adjustment

      | :sl:`Gets or sets an adjustment factor for the underline position`
      | :sg:`underline_adjustment -> float`

      Gets or sets a factor which, when positive, is multiplied with the
      face's underline offset to adjust the underline position. A negative
      value turns an underline into a strikethrough or overline. It is
      multiplied with the ascender. Accepted values are between -2.0 and 2.0
      inclusive. A value of 0.5 closely matches Tango underlining. A value of
      1.0 mimics SDL_ttf.

   .. attribute:: fixed_width

      | :sl:`Gets whether the face is fixed-width`
      | :sg:`fixed_width -> bool`

      Read only. Returns whether this Face is a fixed-width (bitmap) face.

      Note that scalable faces whose glyphs are all the same width (i.e.
      monospace ``TTF`` fonts used for programming) are not considered fixed
      width.

   .. attribute:: antialiased

      | :sl:`Face antialiasing mode`
      | :sg:`antialiased -> bool`

      Gets or sets the face's antialiasing mode. This defaults to ``True`` on
      all faces, which are rendered with full 8 bit blending.

      Setting this to ``False`` will enable monochrome rendering. This should
      provide a small speed gain and reduce cache memory size.

   .. attribute:: kerning

      | :sl:`Character kerning mode`
      | :sg:`kerning -> bool`

      Gets or sets the face's kerning mode. This defaults to False on all
      faces, which will be rendered by default without kerning.

      Setting this to true will change all rendering methods to do kerning
      between character pairs for surface size calculation and all
      render operations.

   .. attribute:: vertical

      | :sl:`Face vertical mode`
      | :sg:`vertical -> bool`

      Gets or sets whether the face is a vertical face such as faces in fonts
      representing Kanji glyphs or other styles of vertical writing.

      Changing this attribute will cause the face to be rendering vertically,
      and affects all other methods which manage glyphs or text layouts to use
      vertical metrics accordingly.

      Note that the FreeType library doesn't automatically detect whether a
      face contains glyphs which are always supposed to be drawn vertically, so
      this attribute must be set manually by the user.

      Also note that several face formats (specially bitmap based ones) don't
      contain the necessary metrics to draw glyphs vertically, so drawing in
      those cases will give unspecified results.

   .. attribute:: origin

      | :sl:`Face render to text origin mode`
      | :sg:`vertical -> bool`

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

      | :sl:`Enables UCS-4 mode`
      | :sg:`ucs4 -> bool`

      Gets or sets the decoding of Unicode textdecoding. By default, the
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

      | :sl:`Output pixel resolution in dots per inch`
      | :sg:`resolution -> int`

      Gets the pixel size used in scaling face glyphs for this Face instance.

   .. method:: set_transform

      | :sl:`assign a glyph transformation matrix`
      | :sg:`set_transform(xx, xy, yx, yy) -> None`

      Set a transform matrix for the face. If None, no matrix assigned.
      The arguments can be any numeric type that can be converted
      to a double. The matrix is applied after the strong transformation,
      but before oblique and rotation.

   .. method:: delete_transform

      | :sl:`delete a glyph transformation matrix`
      | :sg:`set_transform(xx, xy, yx, yy) -> None`

      Remove the transformation matrix, if any.

   .. method:: get_transform

      | :sl:`return the user assigned transformation matrix, or None`
      | :sg:`get_transform() -> (double, double, double, double) or None`

      Return the transform matrix for the face. If None, no matrix is assigned.
