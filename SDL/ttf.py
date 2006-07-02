#!/usr/bin/env python

'''A companion module to SDL for working with TrueType fonts.

This library is a wrapper around FreeType_ 2.0.

.. _FreeType: http://www.freetype.org
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.dll
import SDL.rwops
import SDL.version
import SDL.video

_dll = SDL.dll.SDL_DLL('SDL_ttf', 'TTF_Linked_Version')

TTF_Linked_Version = _dll.function('TTF_Linked_Version',
    '''Get the version of the dynamically linked SDL_ttf library.

    :rtype: `SDL_version`
    ''',
    args=[],
    arg_types=[],
    return_type=POINTER(SDL.version.SDL_version),
    dereference_return=True,
    require_return=True)

# Opaque type pointer
_TTF_Font = c_void_p 

TTF_Init = _dll.function('TTF_Init',
    '''Initialize the TTF engine
    ''',
    args=[],
    arg_types=[],
    return_type=c_int,
    error_return=-1)

TTF_OpenFont = _dll.function('TTF_OpenFont',
    '''Open a font file and create a font of the specified point size.

    :Parameters:
        `file` : string
            Filename of a Truetype font file to open.
        `ptsize` : int
            Size of the font face, in points.  Type is rendered at 64 DPI.

    :rtype: ``TTF_Font``
    ''',
    args=['file', 'ptsize'],
    arg_types=[c_char_p, c_int],
    return_type=_TTF_Font,
    require_return=True)

TTF_OpenFontIndex = _dll.function('TTF_OpenFontIndex',
    '''Open a font collection file and create a font of the specified point 
    size.

    :Parameters:
        `file` : string
            Filename of a Truetype font file to open.
        `ptsize` : int
            Size of the font face, in points.  Type is rendered at 64 DPI.
        `index` : int
            Zero-based index of the desired font within the file.

    :rtype: ``TTF_Font``
    ''',
    args=['file', 'ptsize', 'index'],
    arg_types=[c_char_p, c_int, c_int],
    return_type=_TTF_Font,
    require_return=True)

TTF_OpenFontRW = _dll.function('TTF_OpenFontRW',
    '''Create a font of the specified point size from a RWops object.

    You can create an SDL_RWops object from any Python file-like object
    with `SDL_RWFromObject`.

    :Parameters:
        `src` : `SDL_RWops`
            Readable RWops object.
        `freesrc` : int
            If non-zero, the source will be closed when the face is closed.
        `ptsize` : int
            Size of the font face, in points.  Type is rendered at 64 DPI.

    :rtype: ``TTF_Font``
    ''',
    args=['src', 'freesrc', 'ptsize'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops), c_int, c_int],
    return_type=_TTF_Font,
    require_return=True)

TTF_OpenFontIndexRW = _dll.function('TTF_OpenFontIndexRW',
    '''Create a font of the specified point size from a RWops object.

    :Parameters:
        `src` : `SDL_RWops`
            Readable RWops object.
        `freesrc` : int
            If non-zero, the source will be closed when the face is closed.
        `ptsize` : int
            Size of the font face, in points.  Type is rendered at 64 DPI.
        `index` : int
            Zero-based index of the desired font within the file.

    :rtype: ``TTF_Font``
    ''',
    args=['src', 'freesrc', 'ptsize', 'index'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops), c_int, c_int, c_int],
    return_type=_TTF_Font,
    require_return=True)


TTF_GetFontStyle = _dll.function('TTF_GetFontStyle',
    '''Get the modified font style.

    Note that the modified style has nothing to do with the underlying
    style properties of the font, and merely reflects what has been
    set in `TTF_SetFontStyle`.

    :Parameters:
     - `font`: ``TTF_Font``

    :rtype: int
    :return: A bitwise combination of ``TTF_STYLE_BOLD``, ``TTF_STYLE_ITALIC``
        and ``TTF_STYLE_UNDERLINE``.
    ''',
    args=['font'],
    arg_types=[_TTF_Font],
    return_type=c_int)

TTF_SetFontStyle = _dll.function('TTF_SetFontStyle',
    '''Set the modified font style.

    This font style is implemented by modifying the font glyphs, and doesn't
    reflect any inherent properties of the TrueType font file.
    
    :Parameters:
        `font` : ``TTF_Font``
            Font object to modify.
        `style` : int
            Bitwise combination of any of ``TTF_STYLE_BOLD``,
            ``TTF_STYLE_ITALIC`` and ``TTF_STYLE_UNDERLINE``.

    ''',
    args=['font', 'style'],
    arg_types=[_TTF_Font, c_int],
    return_type=None)

TTF_FontHeight = _dll.function('TTF_FontHeight',
    '''Get the total height of the font, in pixels.

    :Parameters:
     - `font`: ``TTF_Font``

    :rtype: int
    ''',
    args=['font'],
    arg_types=[_TTF_Font],
    return_type=c_int)

TTF_FontAscent = _dll.function('TTF_FontAscent',
    '''Get the ascent of the font, in pixels.

    This is the offset from the baseline to the top of the font.

    :Parameters:
     - `font`: ``TTF_Font``

    :rtype: int
    ''',
    args=['font'],
    arg_types=[_TTF_Font],
    return_type=c_int)

TTF_FontDescent = _dll.function('TTF_FontDescent',
    '''Get the descent of the font, in pixels.

    This is the offset from the baseline to the lowest point of the font,
    and is usually negative.

    :Parameters:
     - `font`: ``TTF_Font``

    :rtype: int
    ''',
    args=['font'],
    arg_types=[_TTF_Font],
    return_type=c_int)

TTF_FontLineSkip = _dll.function('TTF_FontLineSkip',
    '''Get the recommended spacing between lines of text, in pixels.

    :Parameters:
     - `font`: ``TTF_Font``

    :rtype: int
    ''',
    args=['font'],
    arg_types=[_TTF_Font],
    return_type=c_int)

TTF_FontFaces = _dll.function('TTF_FontFaces',
    '''Get the number of fonts in the font collection file.

    :Parameters:
     - `font`: ``TTF_Font``

    :rtype: int
    ''',
    args=['font'],
    arg_types=[_TTF_Font],
    return_type=c_long)

TTF_FontFaceIsFixedWidth = _dll.function('TTF_FontFaceIsFixedWidth',
    '''Determine if a font is monospaced or not.

    :Parameters:
     - `font`: ``TTF_Font``

    :rtype: int
    :return: non-zero if monospaced, otherwise zero.
    ''',
    args=['font'],
    arg_types=[_TTF_Font],
    return_type=c_int)

TTF_FontFaceFamilyName = _dll.function('TTF_FontFaceFamilyName',
    '''Get the family name of the font.

    For example, "Times New Roman".

    :Parameters:
     - `font`: ``TTF_Font``

    :rtype: string
    ''',
    args=['font'],
    arg_types=[_TTF_Font],
    return_type=c_char_p)

TTF_FontFaceStyleName = _dll.function('TTF_FontFaceStyleName',
    '''Get the style name of the font.

    For example, "Regular", "Bold", "Italic", etc.

    :Parameters:
     - `font`: ``TTF_Font``

    :rtype: string
    ''',
    args=['font'],
    arg_types=[_TTF_Font],
    return_type=c_char_p)

_TTF_GlyphMetrics = _dll.private_function('TTF_GlyphMetrics',
    arg_types=[_TTF_Font, c_ushort, POINTER(c_int), POINTER(c_int),
               POINTER(c_int), POINTER(c_int), POINTER(c_int)],
    return_type=int,
    error_return=-1)

def TTF_GlyphMetrics(font, ch):
    '''Get the metrics of a glyph.

    The character `ch` is used to look up the glyph metrics
    for the given font.  The metrics returned are:

        minx, maxx, miny, maxy
            Bounding box for the glyph.
        advance
            Horizontal advance for the glyph.

    All metrics are returned in pixels.

    :Parameters:
        `font` : ``TTF_Font``
            Font object to inspect.
        `ch` : string of length 1
            Character to look up

    :rtype: (int, int, int, int, int)
    :return: (minx, maxx, miny, maxy, advance)
    '''
    minx, maxx, miny, maxy, advance = \
        c_int(), c_int(), c_int(), c_int(), c_int()
    _TTF_GlyphMetrics(font, ord(ch), byref(minx), byref(maxx), byref(miny),
                      byref(maxy), byref(advance))
    return minx.value, maxx.value, miny.value, maxy.value, advance.value

_TTF_SizeUTF8 = _dll.private_function('TTF_SizeUTF8',
    arg_types=[_TTF_Font, c_char_p, POINTER(c_int), POINTER(c_int)],
    return_type=c_int, 
    error_return=-1)

def TTF_SizeText(font, text):
    '''Get the dimensions of a rendered string of text, in pixels.

    :Parameters:
     - `font`: ``TTF_Font``
     - `text`: string

    :rtype: (int, int)
    :return: (width, height)
    '''
    w, h = c_int(), c_int()
    _TTF_SizeUTF8(font, text.encode('utf8'), byref(w), byref(h))
    return w.value, h.value

_TTF_RenderUTF8_Solid = _dll.private_function('TTF_RenderUTF8_Solid',
    arg_types=[_TTF_Font, c_char_p, SDL.video.SDL_Color],
    return_type=POINTER(SDL.video.SDL_Surface), 
    require_return=True,
    dereference_return=True)

def TTF_RenderText_Solid(font, text, fg):
    '''Create an 8-bit palettized surface and render the given text at
    the fast quality with the given font and color.

    The palette has 0 as the colorkey, giving it a transparent background,
    with 1 as the text color.

    :Parameters:
       - `font`: ``TTF_Font``
       - `text`: string
       - `fg`: `SDL_Color`

    :rtype: `SDL_Surface`
    '''
    return _TTF_RenderUTF8_Solid(font, text.encode('utf8'), fg)
      
_TTF_RenderGlyph_Solid = _dll.private_function('TTF_RenderGlyph_Solid',
    arg_types=[_TTF_Font, c_ushort, SDL.video.SDL_Color],
    return_type=POINTER(SDL.video.SDL_Surface), 
    require_return=True,
    dereference_return=True)

def TTF_RenderGlyph_Solid(font, ch, fg):
    '''Create an 8-bit palettized surface and render the given character at
    the fast quality with the given font and color.

    The palette has 0 as the colorkey, giving it a transparent background,
    with 1 as the text color.

    The glyph is rendered without any padding or centering in the X direction,
    and aligned normally in the Y direction.

    :Parameters:
       - `font`: ``TTF_Font``
       - `ch`: string of length 1
       - `fg`: `SDL_Color`

    :rtype: `SDL_Surface`
    '''
    return _TTF_RenderGlyph_Solid(font, ord(text), fg)

_TTF_RenderUTF8_Shaded = _dll.private_function('TTF_RenderUTF8_Shaded',
    arg_types=[_TTF_Font, c_char_p, SDL.video.SDL_Color, SDL.video.SDL_Color],
    return_type=POINTER(SDL.video.SDL_Surface), 
    require_return=True,
    dereference_return=True)

def TTF_RenderText_Shaded(font, text, fg, bg):
    '''Create an 8-bit palettized surface and render the given text at
    high quality with the given font and colors.

    The 0 pixel is background, while other pixels have varying degrees of
    the foreground color.

    :Parameters:
       - `font`: ``TTF_Font``
       - `text`: string
       - `fg`: `SDL_Color`
       - `bg`: `SDL_Color`

    :rtype: `SDL_Surface`
    '''
    return _TTF_RenderUTF8_Shaded(font, text.encode('utf8'), fg, bg)
      
_TTF_RenderGlyph_Shaded = _dll.private_function('TTF_RenderGlyph_Shaded',
    arg_types=[_TTF_Font, c_ushort, SDL.video.SDL_Color, SDL.video.SDL_Color],
    return_type=POINTER(SDL.video.SDL_Surface), 
    require_return=True,
    dereference_return=True)

def TTF_RenderGlyph_Shaded(font, ch, fg, bg):
    '''Create an 8-bit palettized surface and render the given character at
    high quality with the given font and color.

    The 0 pixel is background, while other pixels have varying degrees of
    the foreground color.

    The glyph is rendered without any padding or centering in the X direction,
    and aligned normally in the Y direction.

    :Parameters:
       - `font`: ``TTF_Font``
       - `ch`: string of length 1
       - `fg`: `SDL_Color`
       - `bg`: `SDL_Color`

    :rtype: `SDL_Surface`
    '''
    return _TTF_RenderGlyph_Shaded(font, ord(ch), fg, bg)

_TTF_RenderUTF8_Blended = _dll.private_function('TTF_RenderUTF8_Blended',
    arg_types=[_TTF_Font, c_char_p, SDL.video.SDL_Color],
    return_type=POINTER(SDL.video.SDL_Surface), 
    require_return=True,
    dereference_return=True)

def TTF_RenderText_Blended(font, text, fg):
    '''Create a 32-bit ARGB surface and render the given text at
    high quality, using alpha blending to dither the font with the
    given color.

    :Parameters:
       - `font`: ``TTF_Font``
       - `text`: string
       - `fg`: `SDL_Color`

    :rtype: `SDL_Surface`
    '''
    return _TTF_RenderUTF8_Blended(font, text.encode('utf8'), fg)
      
_TTF_RenderGlyph_Blended = _dll.private_function('TTF_RenderGlyph_Blended',
    arg_types=[_TTF_Font, c_ushort, SDL.video.SDL_Color],
    return_type=POINTER(SDL.video.SDL_Surface), 
    require_return=True,
    dereference_return=True)

def TTF_RenderGlyph_Blended(font, ch, fg):
    '''Create a 32-bit ARGB surface and render the given character at
    high quality, using alpha blending to dither the font with the
    given color.

    The glyph is rendered without any padding or centering in the X direction,
    and aligned normally in the Y direction.

    :Parameters:
       - `font`: ``TTF_Font``
       - `ch`: string of length 1
       - `fg`: `SDL_Color`

    :rtype: `SDL_Surface`
    '''
    return _TTF_RenderGlyph_Blended(font, ord(text), fg)

TTF_CloseFont = _dll.function('TTF_CloseFont',
    '''Close an opened font file.

    :Parameters:
      - `font`: ``TTF_Font``

    ''',
    args=['font'],
    arg_types=[_TTF_Font],
    return_type=None)

TTF_Quit = _dll.function('TTF_Quit',
    '''De-initialize the TTF engine.
    ''',
    args=[],
    arg_types=[],
    return_type=None)

TTF_WasInit = _dll.function('TTF_WasInit',
    '''Check if the TTF engine is initialized.

    :rtype: int
    :return: non-zero if initialized, otherwise zero.
    ''',
    args=[],
    arg_types=[],
    return_type=c_int)
