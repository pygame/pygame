.. include:: common.txt

:mod:`pygame.font`
==================

.. module:: pygame.font
   :synopsis: pygame module for loading and rendering fonts

| :sl:`pygame module for loading and rendering fonts`

The font module allows for rendering TrueType fonts into a new Surface object.
It accepts any UCS-2 character ('\u0001' to '\uFFFF'). This module is optional
and requires SDL_ttf as a dependency. You should test that :mod:`pygame.font`
is available and initialized before attempting to use the module.

Most of the work done with fonts are done by using the actual Font objects. The
module by itself only has routines to initialize the module and create Font
objects with ``pygame.font.Font()``.

You can load fonts from the system by using the ``pygame.font.SysFont()``
function. There are a few other functions to help lookup the system fonts.

Pygame comes with a builtin default font. This can always be accessed by
passing None as the font name.

To use the :mod:`pygame.freetype` based :mod:`pygame.ftfont` as
:mod:`pygame.font` define the enviroment variable PYGAME_FREETYPE before the
first import of :mod:`pygame`. :mod:`pygame.ftfont` is a :mod:`pygame.font`
compatible module that passes all but one of the font module unit tests:
it does not have the UCS-2 limitation of the SDL_ttf based font module, so
fails to raise an exception for a code point greater than '\uFFFF'. If
:mod:`pygame.freetype` is unavailable then the SDL_ttf font module will be
loaded instead.

.. function:: init

   | :sl:`initialize the font module`
   | :sg:`init() -> None`

   This method is called automatically by ``pygame.init()``. It initializes the
   font module. The module must be initialized before any other functions will
   work.

   It is safe to call this function more than once.

   .. ## pygame.font.init ##

.. function:: quit

   | :sl:`uninitialize the font module`
   | :sg:`quit() -> None`

   Manually uninitialize SDL_ttf's font system. This is called automatically by
   ``pygame.quit()``.

   It is safe to call this function even if font is currently not initialized.

   .. ## pygame.font.quit ##

.. function:: get_init

   | :sl:`true if the font module is initialized`
   | :sg:`get_init() -> bool`

   Test if the font module is initialized or not.

   .. ## pygame.font.get_init ##

.. function:: get_default_font

   | :sl:`get the filename of the default font`
   | :sg:`get_default_font() -> string`

   Return the filename of the system font. This is not the full path to the
   file. This file can usually be found in the same directory as the font
   module, but it can also be bundled in separate archives.

   .. ## pygame.font.get_default_font ##

.. function:: get_fonts

   | :sl:`get all available fonts`
   | :sg:`get_fonts() -> list of strings`

   Returns a list of all the fonts available on the system. The names of the
   fonts will be set to lowercase with all spaces and punctuation removed. This
   works on most systems, but some will return an empty list if they cannot
   find fonts.

   .. ## pygame.font.get_fonts ##

.. function:: match_font

   | :sl:`find a specific font on the system`
   | :sg:`match_font(name, bold=False, italic=False) -> path`

   Returns the full path to a font file on the system. If bold or italic are
   set to true, this will attempt to find the correct family of font.

   The font name can actually be a comma separated list of font names to try.
   If none of the given names are found, None is returned.

   Example:

   ::

       print pygame.font.match_font('bitstreamverasans')
       # output is: /usr/share/fonts/truetype/ttf-bitstream-vera/Vera.ttf
       # (but only if you have Vera on your system)

   .. ## pygame.font.match_font ##

.. function:: SysFont

   | :sl:`create a Font object from the system fonts`
   | :sg:`SysFont(name, size, bold=False, italic=False) -> Font`

   Return a new Font object that is loaded from the system fonts. The font will
   match the requested bold and italic flags. If a suitable system font is not
   found this will fallback on loading the default pygame font. The font name
   can be a comma separated list of font names to look for.

   .. ## pygame.font.SysFont ##

.. class:: Font

   | :sl:`create a new Font object from a file`
   | :sg:`Font(filename, size) -> Font`
   | :sg:`Font(object, size) -> Font`

   Load a new font from a given filename or a python file object. The size is
   the height of the font in pixels. If the filename is None the Pygame default
   font will be loaded. If a font cannot be loaded from the arguments given an
   exception will be raised. Once the font is created the size cannot be
   changed.

   Font objects are mainly used to render text into new Surface objects. The
   render can emulate bold or italic features, but it is better to load from a
   font with actual italic or bold glyphs. The rendered text can be regular
   strings or unicode.

   .. method:: render

      | :sl:`draw text on a new Surface`
      | :sg:`render(text, antialias, color, background=None) -> Surface`

      This creates a new Surface with the specified text rendered on it. Pygame
      provides no way to directly draw text on an existing Surface: instead you
      must use ``Font.render()`` to create an image (Surface) of the text, then
      blit this image onto another Surface.

      The text can only be a single line: newline characters are not rendered.
      Null characters ('\x00') raise a TypeError. Both Unicode and char (byte)
      strings are accepted. For Unicode strings only UCS-2 characters ('\u0001'
      to '\uFFFF') are recognized. Anything greater raises a UnicodeError. For
      char strings a ``LATIN1`` encoding is assumed. The antialias argument is
      a boolean: if true the characters will have smooth edges. The color
      argument is the color of the text [e.g.: (0,0,255) for blue]. The
      optional background argument is a color to use for the text background.
      If no background is passed the area outside the text will be transparent.

      The Surface returned will be of the dimensions required to hold the text.
      (the same as those returned by Font.size()). If an empty string is passed
      for the text, a blank surface will be returned that is one pixel wide and
      the height of the font.

      Depending on the type of background and antialiasing used, this returns
      different types of Surfaces. For performance reasons, it is good to know
      what type of image will be used. If antialiasing is not used, the return
      image will always be an 8bit image with a two color palette. If the
      background is transparent a colorkey will be set. Antialiased images are
      rendered to 24-bit ``RGB`` images. If the background is transparent a
      pixel alpha will be included.

      Optimization: if you know that the final destination for the text (on the
      screen) will always have a solid background, and the text is antialiased,
      you can improve performance by specifying the background color. This will
      cause the resulting image to maintain transparency information by
      colorkey rather than (much less efficient) alpha values.

      If you render '\n' a unknown char will be rendered. Usually a rectangle.
      Instead you need to handle new lines yourself.

      Font rendering is not thread safe: only a single thread can render text
      at any time.

      .. ## Font.render ##

   .. method:: size

      | :sl:`determine the amount of space needed to render text`
      | :sg:`size(text) -> (width, height)`

      Returns the dimensions needed to render the text. This can be used to
      help determine the positioning needed for text before it is rendered. It
      can also be used for wordwrapping and other layout effects.

      Be aware that most fonts use kerning which adjusts the widths for
      specific letter pairs. For example, the width for "ae" will not always
      match the width for "a" + "e".

      .. ## Font.size ##

   .. method:: set_underline

      | :sl:`control if text is rendered with an underline`
      | :sg:`set_underline(bool) -> None`

      When enabled, all rendered fonts will include an underline. The underline
      is always one pixel thick, regardless of font size. This can be mixed
      with the bold and italic modes.

      .. ## Font.set_underline ##

   .. method:: get_underline

      | :sl:`check if text will be rendered with an underline`
      | :sg:`get_underline() -> bool`

      Return True when the font underline is enabled.

      .. ## Font.get_underline ##

   .. method:: set_bold

      | :sl:`enable fake rendering of bold text`
      | :sg:`set_bold(bool) -> None`

      Enables the bold rendering of text. This is a fake stretching of the font
      that doesn't look good on many font types. If possible load the font from
      a real bold font file. While bold, the font will have a different width
      than when normal. This can be mixed with the italic and underline modes.

      .. ## Font.set_bold ##

   .. method:: get_bold

      | :sl:`check if text will be rendered bold`
      | :sg:`get_bold() -> bool`

      Return True when the font bold rendering mode is enabled.

      .. ## Font.get_bold ##

   .. method:: set_italic

      | :sl:`enable fake rendering of italic text`
      | :sg:`set_italic(bool) -> None`

      Enables fake rendering of italic text. This is a fake skewing of the font
      that doesn't look good on many font types. If possible load the font from
      a real italic font file. While italic the font will have a different
      width than when normal. This can be mixed with the bold and underline
      modes.

      .. ## Font.set_italic ##

   .. method:: metrics

      | :sl:`Gets the metrics for each character in the pased string.`
      | :sg:`metrics(text) -> list`

      The list contains tuples for each character, which contain the minimum
      ``X`` offset, the maximum ``X`` offset, the minimum ``Y`` offset, the
      maximum ``Y`` offset and the advance offset (bearing plus width) of the
      character. [(minx, maxx, miny, maxy, advance), (minx, maxx, miny, maxy,
      advance), ...]. None is entered in the list for each unrecognized
      character.

      .. ## Font.metrics ##

   .. method:: get_italic

      | :sl:`check if the text will be rendered italic`
      | :sg:`get_italic() -> bool`

      Return True when the font italic rendering mode is enabled.

      .. ## Font.get_italic ##

   .. method:: get_linesize

      | :sl:`get the line space of the font text`
      | :sg:`get_linesize() -> int`

      Return the height in pixels for a line of text with the font. When
      rendering multiple lines of text this is the recommended amount of space
      between lines.

      .. ## Font.get_linesize ##

   .. method:: get_height

      | :sl:`get the height of the font`
      | :sg:`get_height() -> int`

      Return the height in pixels of the actual rendered text. This is the
      average size for each glyph in the font.

      .. ## Font.get_height ##

   .. method:: get_ascent

      | :sl:`get the ascent of the font`
      | :sg:`get_ascent() -> int`

      Return the height in pixels for the font ascent. The ascent is the number
      of pixels from the font baseline to the top of the font.

      .. ## Font.get_ascent ##

   .. method:: get_descent

      | :sl:`get the descent of the font`
      | :sg:`get_descent() -> int`

      Return the height in pixels for the font descent. The descent is the
      number of pixels from the font baseline to the bottom of the font.

      .. ## Font.get_descent ##

   .. ## pygame.font.Font ##

.. ## pygame.font ##
