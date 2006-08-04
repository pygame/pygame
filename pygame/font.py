#!/usr/bin/env python

'''Pygame module for loading and rendering fonts.

The font module allows for rendering TrueType fonts into a new Surface
object. This module is optional and requires SDL.ttf as a dependency. You
should test that pygame.font is available and initialized before attempting
to use the module.
 
Most of the work done with fonts are done by using the actual Font objects.
The module by itself only has routines to initialize the module and create
Font objects with pygame.font.Font().
 
You can load fonts from the system by using the pygame.font.SysFont()
function. There are a few other functions to help lookup the system fonts.

Pygame comes with a builtin default font. This can always be accessed by
passing None as the font name.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from SDL import *
from SDL.ttf import *

import pygame.base
import pygame.pkgdata

_font_initialized = 0
_font_defaultname = 'freesansbold.ttf'

def __PYGAMEinit__():
    global _font_initialized
    if not _font_initialized: 
        pygame.base.register_quit(_font_autoquit)

        try:
            TTF_Init()
            _font_initialized = 1
        except:
            pass
    return _font_initialized

def _font_autoquit():
    global _font_initialized
    if _font_initialized:
        _font_initialized = 0
        TTF_Quit()

def init():
    '''Initialize the font module.

    This method is called automatically by pygame.init().  It initializes the
    font module. The module must be initialized before any other functions
    will work.

    It is safe to call this function more than once.
    '''
    __PYGAMEinit__()

def quit():
    '''Uninitialize the font module.

    Manually uninitialize SDL_ttf's font system. This is called automatically
    by pygame.quit().

    It is safe to call this function even if font is currently not
    initialized.
    '''
    _font_autoquit()

def get_init():
    '''Determine if the font module is initialized.

    :rtype: bool
    '''
    return _font_initialized

def get_default_font():
    '''Get the filename of the default font.

    Return the filename of the system font. This is not the full path to the
    file.  This file can usually be found in the same directory as the font
    module, but it can also be bundled in separate archives.

    :rtype: str
    '''
    return _font_defaultname

class Font(object):
    __slots__ = ['_font', '_rw']

    def __init__(self, file, size):
        '''Create a new Font object from a file.

        Load a new font from a given filename or a python file object. The
        size is the height of the font in pixels. If the filename is None the
        Pygame default font will be loaded. If a font cannot be loaded from
        the arguments given an exception will be raised. Once the font is
        created the size cannot be changed. 

        Font objects are mainly used to render text into new Surface objects.
        The render can emulate bold or italic features, but it is better to
        load from a font with actual italic or bold glyphs. The rendered text
        can be regular strings or unicode.

        :Parameters:
            `file` : str or file-like object
                Optional filename of font.
            `size` : int
                Size of font, in points.

        '''
        if not _font_initialized:
            raise pygame.base.error, 'font not initialized'

        if not file:
            file = pygame.pkgdata.getResource(_font_defaultname)
            size *= 0.6875  # XXX Peter's phone number??

        size = int(max(1, size))

        if hasattr(file, 'read'):
            rw = SDL_RWFromObject(file)
            # Keep the RWops around, the callbacks will be used long for
            # the duration of this Font instance.
            self._rw = rw
            font = TTF_OpenFontRW(rw, 1, size)
        else:
            font = TTF_OpenFont(file, size)
        self._font = font

    def __del__(self):
        if self._font and _font_initialized:
            TTF_CloseFont(self._font)

    def render(self, text, antialias, color, background=None):
        '''Draw text on a new Surface.

        This creates a new Surface with the specified text rendered on it.
        Pygame provides no way to directly draw text on an existing Surface;
        instead you must use Font.render() to create an image (Surface) of the
        text, then blit this image onto another Surface.

        The text can only be a single line: newline characters are not
        rendered.  The antialias argument is a boolean: if true the characters
        will have smooth edges. The color argument is the color of the text
        [e.g.: (0,0,255) for blue]. The optional background argument is a
        color to use for the text background. If no background is passed the
        area outside the text will be transparent.

        The Surface returned will be of the dimensions required to hold the
        text.  (the same as those returned by Font.size()).  If an empty
        string is passed for the text, a blank surface will be returned that
        is one pixel wide and the height of the font.

        Depending on the type of background and antialiasing used, this
        returns different types of Surfaces. For performance reasons, it is
        good to know what type of image will be used. If antialiasing is not
        used, the return image will always be an 8bit image with a two color
        palette. If the background is transparent a colorkey will be set.
        Antialiased images are rendered to 24-bit RGB images. If the
        background is transparent a pixel alpha will be included.

        Optimization: if you know that the final destination for the text (on
        the screen) will always have a solid background, and the text is
        antialiased, you can improve performance by specifying the background
        color. This will cause the resulting image to maintain transparency
        information by colorkey rather than (much less efficient) alpha
        values.

        If you render '\\n' a unknown char will be rendered.  Usually a
        rectangle.  Instead you need to handle new lines yourself.

        Font rendering is not thread safe: only a single thread can render
        text any time.
        
        :Parameters:
            `text` : str or unicode
                Text to render
            `antialias` : bool
                If True, apply antialiasing to glyphs.
            `color` : (int, int, int)
                RGB color of glyphs.
            `background` : (int, int, int)
                Optional RGB color of background.

        :rtype: `Surface`
        '''
        font = self._font
        foreground = SDL_Color(*pygame.base._rgba_from_obj(color))
        if background:
            background = SDL_Color(*pygame.base._rgba_from_obj(background))

        if not text:
            # Pygame returns a 1 pixel wide surface if given empty string.
            height = TTF_FontHeight(font)
            surf = SDL_CreateRGBSurface(SDL_SWSURFACE, 1, height, 32,
                                        0x00ff0000, 0x0000ff00, 0x000000ff, 0)
            if background:
                c = SDL_MapRGB(surf.format, background.r, background.g,
                               background.b)
                SDL_FillRect(surf, None, c)
            else:
                SDL_SetColorKey(surf, SDL_SRCCOLORKEY, 0)
        elif antialias:
            if not background:
                surf = TTF_RenderText_Blended(font, text, foreground)
            else:
                surf = TTF_RenderText_Shaded(font, text, foreground, background)
        else:
            surf = TTF_RenderText_Solid(font, text, foreground)

        if text and not antialias and background:
            # Add color key
            SDL_SetColorKey(surf, 0, 0)
            surf.format.palette.colors[0].r = background.r
            surf.format.palette.colors[0].g = background.g
            surf.format.palette.colors[0].b = background.b

        return pygame.surface.Surface(surf=surf)

    def size(self, text):
        '''Determine the amount of space needed to render text.

        Returns the dimensions needed to render the text. This can be used to
        help determine the positioning needed for text before it is rendered.
        It can also be used for wordwrapping and other layout effects.

        Be aware that most fonts use kerning which adjusts the widths for
        specific letter pairs. For example, the width for "T." will not always
        match the width for "T" + ".".
        
        :Parameters:
            `text` : str or unicode
                Text to measure

        :rtype: int, int
        :return: width, height
        '''
        return TTF_SizeText(self._font, text)

    def set_underline(self, underline):
        '''Control if text is rendered with an underline.

        When enabled, all rendered fonts will include an underline. The
        underline is always one pixel thick, regardless of font size. This can
        be mixed with the bold and italic modes.
        
        :Parameters:
            `underline` : bool
                If True, the text will be rendered with an underline.

        '''
        style = TTF_GetFontStyle(self._font)
        if underline:
            style |= TTF_STYLE_UNDERLINE
        else:
            style &= ~TTF_STYLE_UNDERLINE
        TTF_SetFontStyle(self._font, style)

    def get_underline(self):
        '''Check if text will be rendered with an underline.

        :rtype: bool
        '''
        return TTF_GetFontStyle(self._font) & TTF_STYLE_UNDERLINE != 0

    def set_bold(self, bold):
        '''Enable fake rendering of bold text.

        Enables the bold rendering of text. This is a fake stretching of the
        font that doesn't look good on many font types. If possible load the
        font from a real bold font file. While bold, the font will have a
        different width than when normal. This can be mixed with the italic
        and underline modes.
        
        :Parameters:
            `bold` : bool
                If True, the text will be rendered with a heavier pen.

        '''
        style = TTF_GetFontStyle(self._font)
        if bold:
            style |= TTF_STYLE_BOLD
        else:
            style &= ~TTF_STYLE_BOLD
        TTF_SetFontStyle(self._font, style)

    def get_bold(self):
        '''Check if text will be rendered bold.

        :rtype: bool
        '''
        return TTF_GetFontStyle(self._font) & TTF_STYLE_BOLD != 0

    def set_italic(self, italic):
        '''Enable fake rendering of italic text.

        Enables fake rendering of italic text. This is a fake skewing of the
        font that doesn't look good on many font types. If possible load the
        font from a real italic font file. While italic the font will have a
        different width than when normal. This can be mixed with the bold and
        underline modes.
        
        :Parameters:
            `italic` : bool
                If True, the text will be rendered at an oblique angle.

        '''
        style = TTF_GetFontStyle(self._font)
        if italic:
            style |= TTF_STYLE_ITALIC
        else:
            style &= ~TTF_STYLE_ITALIC
        TTF_SetFontStyle(self._font, style)

    def get_italic(self):
        '''Check if the text will be rendered italic.

        :rtype: bool 
        '''
        return TTF_GetFontStyle(self._font) & TTF_STYLE_ITALIC != 0

    def get_linesize(self):
        '''Get the line space of the font text.

        Return the height in pixels for a line of text with the font. When
        rendering multiple lines of text this is the recommended amount of
        space between lines.

        :rtype: int
        '''
        return TTF_FontLineSkip(self._font)

    def get_height(self):
        '''Get the height of the font.

        Return the height in pixels of the actual rendered text. This is the
        average size for each glyph in the font.

        :rtype: int
        '''
        return TTF_FontHeight(self._font)

    def get_ascent(self):
        '''Get the ascent of the font.

        Return the height in pixels for the font ascent. The ascent is the
        number of pixels from the font baseline to the top of the font.
        
        :rtype: int
        '''
        return TTF_FontAscent(self._font)

    def get_descent(self):
        '''Get the descent of the font.

        Return the height in pixels for the font descent. The descent is the
        number of pixels from the font baseline to the bottom of the font.
        
        :rtype: int
        '''
        return TTF_FontDescent(self._font)

FontType = Font
