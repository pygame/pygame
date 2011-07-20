/*
  pygame - Python Game Library
  Copyright (C) 2009 Vicent Marti

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/

#define PYGAME_FREETYPE_INTERNAL

#include "ft_wrap.h"
#include FT_MODULE_H
#include FT_OUTLINE_H

#define SLANT_FACTOR    0.22
FT_Matrix PGFT_SlantMatrix = 
{
    (1 << 16),  (FT_Fixed)(SLANT_FACTOR * (1 << 16)),
    0,          (1 << 16) 
};

static int _PGFT_Render_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
                                 FontText *text, const FontRenderMode *render,
                                 FontColor *fg_color, FontSurface *surface);

int PGFT_CheckStyle(FT_UInt32 style)
{
    const FT_UInt32 max_style =
        FT_STYLE_NORMAL |
        FT_STYLE_BOLD   |
        FT_STYLE_OBLIQUE |
        FT_STYLE_UNDERLINE;

    return (style > max_style);
}

int 
PGFT_BuildRenderMode(FreeTypeInstance *ft, 
        PyFreeTypeFont *font, FontRenderMode *mode, 
        int pt_size, int style, int rotation)
{
    int angle;

    if (pt_size == -1)
    {
        if (font->ptsize == -1)
        {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typeface");
            return -1;
        }

        pt_size = font->ptsize;
    }

    if (pt_size <= 0)
    {
        RAISE(PyExc_ValueError, "Invalid point size for font.");
        return -1;
    }

    mode->pt_size = (FT_UInt16)pt_size;

    if (style == FT_STYLE_DEFAULT)
    {
        mode->style = (FT_Byte)font->style;
    }
    else
    {
        if (PGFT_CheckStyle((FT_UInt32)style) != 0)
        {
            RAISE(PyExc_ValueError, "Invalid style value");
            return -1;
        }

        mode->style = (FT_Byte)style;
    }

    mode->render_flags = FT_RFLAG_DEFAULTS;

    if (font->vertical)
        mode->render_flags |= FT_RFLAG_VERTICAL;

    if (font->antialias)
        mode->render_flags |= FT_RFLAG_ANTIALIAS;

    angle = rotation % 360;
    while (angle < 0) angle += 360;
    mode->rotation_angle = PGFT_INT_TO_16(angle);

    return 0;
}


/*********************************************************
 *
 * Rendering on SDL-specific surfaces
 *
 *********************************************************/
#ifdef HAVE_PYGAME_SDL_VIDEO
int PGFT_Render_ExistingSurface(
    FreeTypeInstance *ft, PyFreeTypeFont *font,
    const FontRenderMode *render, PGFT_String *text, SDL_Surface *surface,
    int x, int y, FontColor *fgcolor, FontColor *bgcolor,
    int *_width, int *_height, FontMetrics *metrics)
{
    static const FontRenderPtr __SDLrenderFuncs[] =
    {
        NULL,
        __render_glyph_RGB1,
        __render_glyph_RGB2,
        __render_glyph_RGB3,
        __render_glyph_RGB4
    };

    static const FontRenderPtr __MONOrenderFuncs[] =
    {
        NULL,
        __render_glyph_MONO1,
        __render_glyph_MONO2,
        __render_glyph_MONO3,
        __render_glyph_MONO4
    };

    static const FontFillPtr __RGBfillFuncs[] =
    {
        NULL,
        __fill_glyph_RGB1,
        __fill_glyph_RGB2,
        __fill_glyph_RGB3,
        __fill_glyph_RGB4
    };

    int         locked = 0;
    int         width, height;

    FontSurface font_surf;
    FontText    *font_text;

    if (PGFT_String_GET_LENGTH(text) == 0)
    {
        /* No rendering */
        *_width = 0;
        *_height = PGFT_Face_GetHeight(ft, font);
        return 0;
    }

    if (SDL_MUSTLOCK(surface))
    {
        if (SDL_LockSurface(surface) == -1)
        {
            SDL_FreeSurface(surface);
            RAISE(PyExc_SDLError, SDL_GetError());
            return -1;
        }
        locked = 1;
    }

    /* build font text */
    font_text = PGFT_LoadFontText(ft, font, render, text);

    if (!font_text)
    {
        if (locked)
            SDL_UnlockSurface(surface);
        return -1;
    }

    if (PGFT_GetSurfaceSize(ft, font, render, font_text, &width, &height) != 0)
    {
        if (locked)
            SDL_UnlockSurface(surface);
        return -1;
    }

    if (width <= 0 || height <= 0)
    {
        /* Nothing more to do. */
        if (locked)
            SDL_UnlockSurface(surface);
        *_width = 0;
        *_height = PGFT_Face_GetHeight(ft, font);
        return 0;
    }
    
    /*
     * Setup target surface struct
     */
    font_surf.buffer = surface->pixels;
    font_surf.offset.x = PGFT_INT_TO_6(x);
    font_surf.offset.y = PGFT_INT_TO_6(y);
    if (font->origin)
    {
        x -= PGFT_TRUNC(PGFT_CEIL(font_text->offset.x));
        y -= PGFT_TRUNC(PGFT_CEIL(font_text->offset.y));
    }
    else
    {
        font_surf.offset.x += font_text->offset.x;
        font_surf.offset.y += font_text->offset.y;
    }

    font_surf.width = surface->w;
    font_surf.height = surface->h;
    font_surf.pitch = surface->pitch / surface->format->BytesPerPixel;

    font_surf.format = surface->format;

    font_surf.render_gray = __SDLrenderFuncs[surface->format->BytesPerPixel];
    font_surf.render_mono = __MONOrenderFuncs[surface->format->BytesPerPixel];
    font_surf.fill = __RGBfillFuncs[surface->format->BytesPerPixel];

    /* 
     * if bg color exists, paint background 
     */
    if (bgcolor)
    {
        if (bgcolor->a == 0xFF)
        {
            SDL_Rect    bg_fill; 
            FT_UInt32   fillcolor;

            fillcolor = SDL_MapRGBA(surface->format, 
                    bgcolor->r, bgcolor->g, bgcolor->b, bgcolor->a);

            bg_fill.x = (FT_Int16)x;
            bg_fill.y = (FT_Int16)y;
            bg_fill.w = (FT_UInt16)width;
            bg_fill.h = (FT_UInt16)height;

            SDL_FillRect(surface, &bg_fill, fillcolor);
        }
        else
        {
            font_surf.fill(x, y, width, height, &font_surf, bgcolor);
        }
    }

    /*
     * Render!
     */
    if (_PGFT_Render_INTERNAL(ft, font, font_text, render, fgcolor, &font_surf))
    {
        if (locked)
            SDL_UnlockSurface(surface);
        return -1;
    }

    *_width = width;
    *_height = height;
    metrics->bearing_rotated.x = font_text->offset.x;
    metrics->bearing_rotated.y = font_text->offset.y;
    metrics->advance_rotated.x = font_text->advance.x;
    metrics->advance_rotated.y = font_text->advance.y;

    if (locked)
        SDL_UnlockSurface(surface);

    return 0;
}

SDL_Surface *PGFT_Render_NewSurface(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    const FontRenderMode *render, PGFT_String *text,
    FontColor *fgcolor, FontColor *bgcolor, int *_width, int *_height)
{
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    FT_UInt32 rmask = 0xff000000;
    FT_UInt32 gmask = 0x00ff0000;
    FT_UInt32 bmask = 0x0000ff00;
    FT_UInt32 amask = 0x000000ff;
#else
    FT_UInt32 rmask = 0x000000ff;
    FT_UInt32 gmask = 0x0000ff00;
    FT_UInt32 bmask = 0x00ff0000;
    FT_UInt32 amask = 0xff000000;
#endif
    int locked = 0;
    FT_UInt32 fillcolor;
    SDL_Surface *surface = NULL;
    FT_UInt32 surface_flags = SDL_SWSURFACE | SDL_SRCALPHA;

    FontSurface font_surf;
    FontText *font_text;
    int width, height;

    if (PGFT_String_GET_LENGTH(text) == 0)
    {
        /* Empty surface */
        *_width = 0;
        *_height = PGFT_Face_GetHeight(ft, font);
        return SDL_CreateRGBSurface(SDL_SWSURFACE, 0, *_height, 32,
                                    rmask, gmask, bmask, amask);
    }

    /* build font text */
    font_text = PGFT_LoadFontText(ft, font, render, text);

    if (!font_text)
        return NULL;

    if (PGFT_GetSurfaceSize(ft, font, render, font_text, &width, &height) != 0)
        return NULL;
    if (height <= 0)
    {
        height = PGFT_Face_GetHeight(ft, font);
    }
        
    surface = SDL_CreateRGBSurface(surface_flags, width, height,
				   32, rmask, gmask, bmask, amask);
    if (!surface)
    {
        PyErr_NoMemory(); /* Everything else should be Okay */
        return NULL;
    }

    if (width <= 0)
    {
        /* Nothing more to do. */
        *_width = 0;
        *_height = height;
        return surface;
    }

    if (SDL_MUSTLOCK(surface))
    {
        if (SDL_LockSurface(surface) == -1)
        {
            RAISE(PyExc_SDLError, SDL_GetError());
            SDL_FreeSurface(surface);
            return NULL;
        }
        locked = 1;
    }

    font_surf.buffer = surface->pixels;
    font_surf.offset.x = font_text->offset.x;
    font_surf.offset.y = font_text->offset.y;

    font_surf.width = surface->w;
    font_surf.height = surface->h;
    font_surf.pitch = surface->pitch / sizeof(FT_UInt32);

    font_surf.format = surface->format;
    font_surf.render_gray = __render_glyph_RGB4;
    font_surf.render_mono = __render_glyph_MONO4;
    font_surf.fill = __fill_glyph_RGB4;

    /*
     * Fill our texture with the required bg color
     */
    if (bgcolor)
    {
        fillcolor = SDL_MapRGBA(surface->format, 
                bgcolor->r, bgcolor->g, bgcolor->b, bgcolor->a);
    }
    else
    {
        fillcolor = SDL_MapRGBA(surface->format, 0, 0, 0, 0);
    }

    SDL_FillRect(surface, NULL, fillcolor);

    /*
     * Render the text!
     */
    if (_PGFT_Render_INTERNAL(ft, font, font_text, render,
                              fgcolor, &font_surf))
    {
        SDL_FreeSurface(surface);
        return NULL;
    }

    *_width = width;
    *_height = height;

    if (locked)
        SDL_UnlockSurface(surface);

    return surface;
}
#endif  /* #ifdef HAVE_PYGAME_SDL_VIDEO */




/*********************************************************
 *
 * Rendering on generic arrays
 *
 *********************************************************/

PyObject *PGFT_Render_PixelArray(FreeTypeInstance *ft, PyFreeTypeFont *font,
    const FontRenderMode *render, PGFT_String *text, int *_width, int *_height)
{
    FT_Byte *buffer = NULL;
    PyObject *array = NULL;
    FontSurface surf;
    int width, height;

    FontText *font_text;
    int array_size;

    if (PGFT_String_GET_LENGTH(text) == 0)
    {
        /* Empty array */
        *_width = 0;
        *_height = PGFT_Face_GetHeight(ft, font);
        return Bytes_FromStringAndSize("", 0);
    }

    /* build font text */
    font_text = PGFT_LoadFontText(ft, font, render, text);

    if (!font_text)
        return NULL;

    if (PGFT_GetSurfaceSize(ft, font, render, font_text, &width, &height) != 0)
        return NULL;

    array_size = width * height;

    if (array_size <= 0)
    {
        /* Empty array */
        *_width = 0;
        *_height = PGFT_Face_GetHeight(ft, font);
        return Bytes_FromStringAndSize("", 0);
    }

    /* Create an uninitialized string whose buffer can be directly set. */
    array = Bytes_FromStringAndSize(NULL, array_size);
    if (array == NULL)
    {
        return NULL;
    }
    buffer = (FT_Byte *)Bytes_AS_STRING(array);

    memset(buffer, 0x00, (size_t)array_size);

    surf.buffer = buffer;
    surf.offset.x = font_text->offset.x;
    surf.offset.y = font_text->offset.y;
    surf.width = surf.pitch = width;
    surf.height = height;

    surf.format = NULL;
    surf.render_gray = __render_glyph_ByteArray;
    surf.render_mono = __render_glyph_ByteArray_MONO;

    if (_PGFT_Render_INTERNAL(ft, font, font_text, render, 0x0, &surf) != 0)
    {
        Py_DECREF(array);
        return NULL;
    }

    *_width = width;
    *_height = height;

    return array;
}




/*********************************************************
 *
 * New rendering algorithm (rotation + veritical drawing)
 *
 *********************************************************/
static int _PGFT_Render_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
                                 FontText *text, const FontRenderMode *render,
                                 FontColor *fg_color, FontSurface *surface)
{
    FT_Pos top;
    FT_Pos left;
    int x;
    int y;
    int n;
    int length = text->length;
    FontGlyph **glyphs = text->glyphs;
    FT_BitmapGlyph image;
    FT_Vector *posns = text->posns;
    FontRenderPtr render_gray = surface->render_gray;
    FontRenderPtr render_mono = surface->render_mono;
    int error = 0;
    

    if (length <= 0)
        return error;
    left = surface->offset.x;
    top = surface->offset.y;
    for (n = 0; n < length; ++n)
    {
        image = glyphs[n]->image;
        x = PGFT_TRUNC(PGFT_CEIL(left + posns[n].x));
        y = PGFT_TRUNC(PGFT_CEIL(top + posns[n].y));
        if (image->bitmap.pixel_mode == FT_PIXEL_MODE_GRAY)
            render_gray(x, y, surface, &(image->bitmap), fg_color);
        else
            render_mono(x, y, surface, &(image->bitmap), fg_color);
    }

    if (text->underline_size > 0)
    {
        surface->fill(
            PGFT_TRUNC(PGFT_CEIL(left - text->offset.x)),
            PGFT_TRUNC(PGFT_CEIL(top + text->underline_pos)),
            text->width, PGFT_TRUNC(PGFT_CEIL(text->underline_size)),
            surface, fg_color);
    }

    if (error)
    {
        RAISE(PyExc_SDLError, "(exception under construction)"
                              " last character unrendered");
    }
    return error;
}
