.. include:: common.txt

:mod:`pygame.freetype`
======================

.. module:: pygame.freetype
   :synopsis: Enhanced Pygame module for loading and rendering fonts

| :sl:`Enhanced Pygame module for loading and rendering fonts`

--- Note that some features may change before a formal release

This module allows for rendering all font formats supported by FreeType, namely
``TTF``, Type1, ``CFF``, OpenType, ``SFNT``, ``PCF``, ``FNT``, ``BDF``, ``PFR``
and Type42 fonts. It can render any UTF-32 character in a font file.

This module is optional, and replaces all of the functionality of the original
'font' module, whilst expanding it. This module depends in no way on the
SDL_ttf library.

You should test that :mod:`pygame.freetype` is initialized before attempting to
use the module; if the module is available and loaded, it will be automatically
initialized by ``pygame.init()``

Most of the work done with fonts is done by using the actual Font objects. The
module by itself only has routines to initialize itself and create Font objects
with ``pygame.freetype.Font()``.

You can load fonts from the system by using the ``pygame.freetype.SysFont()``
function. There are a few other functions to help lookup the system fonts.

Pygame comes with a builtin default font. This can always be accessed by
passing None as the font name to the Font constructor.

.. function:: get_error

   | :sl:`Get the latest error`
   | :sg:`get_error() -> str`

   Returns the description of the last error which occurred in the FreeType
   library, or None if no errors have occurred.

   .. ## pygame.freetype.get_error ##

.. function:: get_version

   | :sl:`Get the FreeType version`
   | :sg:`get_version() -> (int, int, int)`

   Gets the version of the FreeType2 library which was used to build the
   'freetype' module.

   Note that the FreeType module depends on the FreeType 2 library, and will
   not compile with the original FreeType 1.0, hence the first element of the
   tuple will always be 2.

   .. ## pygame.freetype.get_version ##

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
   be given to adjust font scaling.

   .. ## pygame.freetype.init ##

.. function:: quit

   | :sl:`Shuts down the underlying FreeType 2 library.`
   | :sg:`quit() -> None`

   This function de-initializes the 'freetype' module. After calling this
   function, you should not invoke any class, method or function related to the
   'freetype' module as they are likely to fail or might give unpredictable
   results. It is safe to call this function even if the module hasn't been
   initialized yet.

   .. ## pygame.freetype.quit ##

.. function:: was_init

   | :sl:`Returns whether the the FreeType 2 library is initialized.`
   | :sg:`was_init() -> bool`

   Returns whether the the FreeType 2 library is initialized.

   .. ## pygame.freetype.was_init ##

.. function:: get_default_resolution

   | :sl:`Return the default pixel size in dots per inch`
   | :sg:`get_default_resolution() -> long`

   Returns the default pixel size, in dots per inch, as set by :func:`init`.

   .. ## pygame.freetype.get_default_resolution

.. class:: Font

   | :sl:`Creates a new Font from a supported font file.`
   | :sg:`Font(file, style=STYLE_NONE, ptsize=-1, face_index=0, resolution=0) -> Font`

   'file' can be either a string representing the font's filename, a file-like
   object containing the font, or None; in this last case the default, built-in
   font will be used.

   Optionally, a \*ptsize* argument may be specified to set the default size in
   points which will be used to render the font. Such size can also be
   specified manually on each method call. Because of the way the caching
   system works, specifying a default size on the constructor doesn't imply a
   performance gain over manually passing the size on each function call.

   If the font file has more than one face, the \*index* argument may be
   specified to specify which face index to load. Defaults to 0; font loading
   will fail if the given index is not contained in the font.

   The 'style' argument will set the default style (italic, underline, bold)
   used to draw this font. This style may be overriden on any ``Font.render()``
   call.

   The optional resolution argument sets the pixel size, in dots per inche,
   to use for scaling glyphs for this Font instance. If 0 then the default
   module value, set by :meth:`freetype.init`, is used.

   .. attribute:: name

      | :sl:`Gets the name of the font face.`
      | :sg:`name -> string`

      Read only. Returns the real (long) name of the font type face, as
      specified on the font file.

      .. ## Font.name ##

   .. method:: get_size

      | :sl:`Gets the size of rendered text`
      | :sg:`get_size(text, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> (int, int)`

      Gets the size in pixels which 'text' will occupy when rendered using this
      Font. The calculations will take into account the font's default style
      (e.g. underlined fonts take extra height for the underline), or the style
      may be overridden by the 'style' parameter.

      Returns a tuple containing the width and height of the text's bounding
      box.

      The calculations are done using the font's default size in points,
      without any rotation, and taking into account fonts which are set to be
      drawn vertically via the :meth:`Font.vertical` attribute. Optionally you
      may specify another point size to use via the 'ptsize' argument, or a
      text rotation via the 'rotation' argument.

      If text is a char (byte) string, then its encoding is assumed to be
      ``LATIN1``.

      .. ## Font.get_size ##

   .. method:: get_metrics

      | :sl:`Gets glyph metrics for the font's characters`
      | :sg:`get_metrics(text, ptsize=default) -> [(...), ...]`

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

      The metrics are adjusted for the current rotation, bold, and italics
      settings.

      If text is a char (byte) string, then its encoding is assumed to be
      ``LATIN1``.

      .. ## Font.get_metrics ##

   .. attribute:: height

      | :sl:`Gets the height of the Font`
      | :sg:`height -> int`

      Read only. Gets the height of the Font. This is the average value of all
      glyphs in the font.

      .. ## Font.height ##

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

      If 'None' is passed instead of a destination sequence, a new 32 bit
      :mod:`pygame.Surface` will be created with the required size to contain
      the drawn text, and using \*bgcolor* as its background color. If a
      background color is not available, the surface will be filled with zero
      alpha opacity.

      The return value is a tuple: the target surface and the bounding
      rectangle giving the size and position of the rendered text within the
      surface.

      If an empty string is passed for text then the returned Rect is zero
      width and the height of the font. If dest is None the returned surface is
      the same dimensions as the boundary rect. The rect will test False.

      The rendering is done using the font's default size in points and its
      default style, without any rotation, and taking into account fonts which
      are set to be drawn vertically via the :meth:`Font.vertical` attribute.
      Optionally you may specify another point size to use via the 'ptsize'
      argument, a text rotation via the 'rotation' argument, or a new text
      style via the 'style' argument.

      If text is a char (byte) string, then its encoding is assumed to be
      ``LATIN1``.

      .. ## Font.render ##

   .. method:: render_raw

      | :sl:`Renders text as a string of bytes`
      | :sg:`render_raw(text, ptsize=default) -> (bytes, (int, int))`

      Like ``Font.render(None, ...)`` but the tuple returned is an 8 bit
      monochrome string of bytes and its size. The forground color is 255, the
      background 0, useful as an alpha mask for a foreground pattern.

      .. ## Font.render_raw ##

   .. attribute:: style

      | :sl:`Gets or sets the font's style`
      | :sg:`style -> int`

      Gets or sets the default style of the Font. This default style will be
      used for all text rendering and size calculations unless overriden
      specifically in the \`render()` or \`get_size()` calls. The style value
      may be a bitwise ``OR`` of one or more of the following constants:

      ::

          STYLE_NONE
          STYLE_UNDERLINE
          STYLE_ITALIC
          STYLE_BOLD

      These constants may be found on the FreeType constants module.
      Optionally, the default style can be modified or obtained accessing the
      individual style attributes (underline, italic, bold).

      .. ## Font.style ##

   .. attribute:: underline

      | :sl:`Gets or sets the font's underline style`
      | :sg:`underline -> bool`

      Gets or sets whether the font will be underlined when drawing text. This
      default style value will be used for all text rendering and size
      calculations unless overriden specifically in the \`render()` or
      \`get_size()` calls, via the 'style' parameter.

      .. ## Font.underline ##

   .. attribute:: bold

      | :sl:`Gets or sets the font's bold style`
      | :sg:`bold -> bool`

      Gets or sets whether the font will be bold when drawing text. This
      default style value will be used for all text rendering and size
      calculations unless overriden specifically in the \`render()` or
      \`get_size()` calls, via the 'style' parameter.

      .. ## Font.bold ##

   .. attribute:: italic

      | :sl:`Gets or sets the font's italic style`
      | :sg:`italic -> bool`

      Gets or sets whether the font will be in italics when drawing text. This
      default style value will be used for all text rendering and size
      calculations unless overriden specifically in the \`render()` or
      \`get_size()` calls, via the 'style' parameter.

      .. ## Font.italic ##

   .. attribute:: fixed_width

      | :sl:`Gets whether the font is fixed-width`
      | :sg:`fixed_width -> bool`

      Read only. Returns whether this Font is a fixed-width (bitmap) font.

      Note that scalable fonts whose glyphs are all the same width (i.e.
      monospace ``TTF`` fonts used for programming) are not considered fixed
      width.

      .. ## Font.fixed_width ##

   .. attribute:: antialiased

      | :sl:`Font antialiasing mode`
      | :sg:`antialiased -> bool`

      Gets or sets the font's antialiasing mode. This defaults to True on all
      fonts, which will be rendered by default antialiased.

      Setting this to true will change all rendering methods to use glyph
      bitmaps without antialiasing, which supposes a small speed gain and a
      significant memory gain because of the way glyphs are cached.

      .. ## Font.antialiased ##

   .. attribute:: kerning

      | :sl:`Character kerning mode`
      | :sg:`kerning -> bool`

      Gets or sets the font's kerning mode. This defaults to False on all
      fonts, which will be rendered by default without kerning.

      Setting this to true will change all rendering methods to do kerning
      between character pairs for surface size calculation and all
      render operations.

      .. ## Font.kerning ##

   .. attribute:: vertical

      | :sl:`Font vertical mode`
      | :sg:`vertical -> bool`

      Gets or sets whether the font is a vertical font such as fonts
      representing Kanji glyphs or other styles of vertical writing.

      Changing this attribute will cause the font to be rendering vertically,
      and affects all other methods which manage glyphs or text layouts to use
      vertical metrics accordingly.

      Note that the FreeType library doesn't automatically detect whether a
      font contains glyphs which are always supposed to be drawn vertically, so
      this attribute must be set manually by the user.

      Also note that several font formats (specially bitmap based ones) don't
      contain the necessary metrics to draw glyphs vertically, so drawing in
      those cases will give unspecified results.

      .. ## Font.vertical ##

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

      .. ## Font.usc4 ##

   .. attribute:: resolution

      | :sl:`Output pixel resolution in dots per inch`
      | :sg:`resolution -> int`

      Gets the pixel size used in scaling font glyphs for this Font instance.

   .. ##  ##

   .. ## pygame.freetype.Font ##

.. ## pygame.freetype ##
