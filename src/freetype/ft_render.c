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

#include "ft_mod.h"
#include "ft_wrap.h"
#include "pgfreetype.h"
#include "pgtypes.h"
#include "freetypebase_doc.h"

#include FT_MODULE_H

#ifdef HAVE_PYGAME_SDL_VIDEO
#   include "surface.h"
#endif

#define BOLD_FACTOR     0.08
#define SLANT_FACTOR    0.22

typedef void (* FontRenderPtr)(int, int, FontSurface *, FT_Bitmap *, PyColor *);

/* blitters */
void __render_glyph_MONO1(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_MONO2(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_MONO3(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_MONO4(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);

void __render_glyph_RGB1(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_RGB2(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_RGB3(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_RGB4(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);

void __render_glyph_ByteArray(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);

void 
PGFT_BuildRenderMode(FontRenderMode *mode, int style, int vertical, int antialias, int rotation)
{
    double      radian;
    FT_Fixed    cosinus;
    FT_Fixed    sinus;
    int         angle;

    angle = rotation % 360;
    while (angle < 0) angle += 360;

    mode->kerning_mode = 1;
    mode->kerning_degree = 0;

    mode->style = (FT_Byte)style;

    mode->vertical = (FT_Byte)vertical;
    mode->hinted = (FT_Byte)1;
    mode->autohint = 0;
    mode->antialias = (FT_Byte)antialias;
    mode->matrix = NULL;
    mode->_rotation_angle = 0;

    if (angle != 0)
    {
        radian  = angle * 3.14159 / 180.0;
        cosinus = (FT_Fixed)(cos(radian) * 65536.0);
        sinus   = (FT_Fixed)(sin(radian) * 65536.0);

        mode->_rotation_matrix.xx = cosinus;
        mode->_rotation_matrix.yx = sinus;
        mode->_rotation_matrix.xy = -sinus;
        mode->_rotation_matrix.yy = cosinus;
        mode->_rotation_angle = (angle << 16);

        mode->matrix = &mode->_rotation_matrix;
    }
}


/*********************************************************
 *
 * Rendering on SDL-specific surfaces
 *
 *********************************************************/
#ifdef HAVE_PYGAME_SDL_VIDEO

#define __MONO_RENDER_INNER_LOOP(_bpp, _code)                       \
    for (j = ry; j < max_y; ++j)                                    \
    {                                                               \
        unsigned char*  _src = src;                                 \
        unsigned char*  _dst = dst;                                 \
        FT_UInt32       val = (FT_UInt32)(*_src++ | 0x100) << shift;\
                                                                    \
        for (i = rx; i < max_x; ++i, _dst += _bpp)                  \
        {                                                           \
            if (val & 0x10000)                                      \
                val = (FT_UInt32)(*_src++ | 0x100);                 \
                                                                    \
            if (val & 0x80)                                         \
            {                                                       \
                _code;                                              \
            }                                                       \
                                                                    \
            val   <<= 1;                                            \
        }                                                           \
                                                                    \
        src += bitmap->pitch;                                       \
        dst += surface->pitch * _bpp;                               \
    }                                                               \

#define _CREATE_MONO_RENDER(_bpp, _getp, _setp, _blendp)                \
    void __render_glyph_MONO##_bpp(int x, int y, FontSurface *surface,  \
            FT_Bitmap *bitmap, PyColor *color)                          \
    {                                                                   \
        const int off_x = (x < 0) ? -x : 0;                             \
        const int off_y = (y < 0) ? -y : 0;                             \
                                                                        \
        const int max_x = MIN(x + bitmap->width, surface->width);       \
        const int max_y = MIN(y + bitmap->rows, surface->height);       \
                                                                        \
        const int rx = MAX(0, x);                                       \
        const int ry = MAX(0, y);                                       \
                                                                        \
        int             i, j, shift;                                    \
        unsigned char*  src;                                            \
        unsigned char*  dst;                                            \
        FT_UInt32       full_color;                                     \
        FT_UInt32 bgR, bgG, bgB, bgA;                                   \
                                                                        \
        src  = bitmap->buffer + (off_y * bitmap->pitch) + (off_x >> 3); \
        dst = (unsigned char *)surface->buffer + (rx * _bpp) +          \
                    (ry * _bpp * surface->pitch);                       \
                                                                        \
        full_color = SDL_MapRGBA(surface->format, (FT_Byte)color->r,    \
                (FT_Byte)color->g, (FT_Byte)color->b, 255);             \
                                                                        \
        shift = off_x & 7;                                              \
                                                                        \
        if (color->a == 0xFF)                                           \
        {                                                               \
            __MONO_RENDER_INNER_LOOP(_bpp,                              \
            {                                                           \
                _setp;                                                  \
            });                                                         \
        }                                                               \
        else if (color->a > 0)                                          \
        {                                                               \
            __MONO_RENDER_INNER_LOOP(_bpp,                              \
            {                                                           \
                FT_UInt32 pixel = (FT_UInt32)_getp;                     \
                                                                        \
                GET_RGB_VALS(                                           \
                        pixel, surface->format,                         \
                        bgR, bgG, bgB, bgA);                            \
                                                                        \
                ALPHA_BLEND(                                            \
                        color->r, color->g, color->b, color->a,         \
                        bgR, bgG, bgB, bgA);                            \
                                                                        \
                _blendp;                                                \
            });                                                         \
        }                                                               \
    }

#define _CREATE_RGB_RENDER(_bpp, _getp, _setp, _blendp)                 \
    void __render_glyph_RGB##_bpp(int x, int y, FontSurface *surface,   \
        FT_Bitmap *bitmap, PyColor *color)                              \
    {                                                                   \
        const int off_x = (x < 0) ? -x : 0;                             \
        const int off_y = (y < 0) ? -y : 0;                             \
                                                                        \
        const int max_x = MIN(x + bitmap->width, surface->width);       \
        const int max_y = MIN(y + bitmap->rows, surface->height);       \
                                                                        \
        const int rx = MAX(0, x);                                       \
        const int ry = MAX(0, y);                                       \
                                                                        \
        FT_Byte *dst = ((FT_Byte*)surface->buffer) + (rx * _bpp) +      \
                        (ry * _bpp * surface->pitch);                   \
        FT_Byte *_dst;                                                  \
                                                                        \
        const FT_Byte *src = bitmap->buffer + off_x +                   \
                                (off_y * bitmap->pitch);                \
        const FT_Byte *_src;                                            \
                                                                        \
        const FT_UInt32 full_color =                                    \
            SDL_MapRGBA(surface->format, (FT_Byte)color->r,             \
                    (FT_Byte)color->g, (FT_Byte)color->b, 255);         \
                                                                        \
        FT_UInt32 bgR, bgG, bgB, bgA;                                   \
        int j, i;                                                       \
                                                                        \
        for (j = ry; j < max_y; ++j)                                    \
        {                                                               \
            _src = src;                                                 \
            _dst = dst;                                                 \
                                                                        \
            for (i = rx; i < max_x; ++i, _dst += _bpp)                  \
            {                                                           \
                FT_UInt32 alpha = (*_src++);                            \
                alpha = (alpha * color->a) / 255;                       \
                                                                        \
                if (alpha == 0xFF)                                      \
                {                                                       \
                    _setp;                                              \
                }                                                       \
                else if (alpha > 0)                                     \
                {                                                       \
                    FT_UInt32 pixel = (FT_UInt32)_getp;                 \
                                                                        \
                    GET_RGB_VALS(                                       \
                            pixel, surface->format,                     \
                            bgR, bgG, bgB, bgA);                        \
                                                                        \
                    ALPHA_BLEND(                                        \
                            color->r, color->g, color->b, alpha,        \
                            bgR, bgG, bgB, bgA);                        \
                                                                        \
                    _blendp;                                            \
                }                                                       \
            }                                                           \
                                                                        \
            dst += surface->pitch * _bpp;                               \
            src += bitmap->pitch;                                       \
        }                                                               \
    }


#define _SET_PIXEL_24   \
    SET_PIXEL24_RGB(_dst, surface->format, color->r, color->g, color->b);

#define _BLEND_PIXEL_24 \
    SET_PIXEL24_RGB(_dst, surface->format, bgR, bgG, bgB);

#define _SET_PIXEL(T) \
    *(T*)_dst = (T)full_color;

#define _BLEND_PIXEL(T) *((T*)_dst) = (T)(                          \
    ((bgR >> surface->format->Rloss) << surface->format->Rshift) |  \
    ((bgG >> surface->format->Gloss) << surface->format->Gshift) |  \
    ((bgB >> surface->format->Bloss) << surface->format->Bshift) |  \
    ((255 >> surface->format->Aloss) << surface->format->Ashift  &  \
     surface->format->Amask)                                        )

#define _BLEND_PIXEL_GENERIC(T) *(T*)_dst = (T)(    \
    SDL_MapRGB(surface->format,                     \
        (FT_Byte)bgR, (FT_Byte)bgG, (FT_Byte)bgB)   )

#define _GET_PIXEL(T)    (*((T*)_dst))

_CREATE_RGB_RENDER(4,  _GET_PIXEL(FT_UInt32),   _SET_PIXEL(FT_UInt32),  _BLEND_PIXEL(FT_UInt32))
_CREATE_RGB_RENDER(3,  GET_PIXEL24(_dst),       _SET_PIXEL_24,          _BLEND_PIXEL_24)
_CREATE_RGB_RENDER(2,  _GET_PIXEL(FT_UInt16),   _SET_PIXEL(FT_UInt16),  _BLEND_PIXEL(FT_UInt16))
_CREATE_RGB_RENDER(1,  _GET_PIXEL(FT_Byte),     _SET_PIXEL(FT_Byte),    _BLEND_PIXEL_GENERIC(FT_Byte))

_CREATE_MONO_RENDER(4,  _GET_PIXEL(FT_UInt32),   _SET_PIXEL(FT_UInt32),  _BLEND_PIXEL(FT_UInt32))
_CREATE_MONO_RENDER(3,  GET_PIXEL24(_dst),       _SET_PIXEL_24,          _BLEND_PIXEL_24)
_CREATE_MONO_RENDER(2,  _GET_PIXEL(FT_UInt16),   _SET_PIXEL(FT_UInt16),  _BLEND_PIXEL(FT_UInt16))
_CREATE_MONO_RENDER(1,  _GET_PIXEL(FT_Byte),     _SET_PIXEL(FT_Byte),    _BLEND_PIXEL_GENERIC(FT_Byte))

int PGFT_Render_ExistingSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
    int font_size, FontRenderMode *render, PyObject *text, PySDLSurface *_surface, 
    int x, int y, PyColor *fgcolor, PyColor *bgcolor, int *_width, int *_height)
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
        __render_glyph_MONO4,
    };

    int         locked = 0;
    int         width, height;

    SDL_Surface *surface;
    FontSurface font_surf;
    FontText    *font_text;

    surface = PySDLSurface_AsSDLSurface(_surface);

    if (SDL_MUSTLOCK(surface))
    {
        if (SDL_LockSurface(surface) == -1)
        {
            _PGFT_SetError(ft, SDL_GetError (), 0);
            SDL_FreeSurface(surface);
            return -1;
        }
        locked = 1;
    }

    /* build font text */
    font_text = PGFT_LoadFontText(ft, font, font_size, render, text);

    if (!font_text)
        return -1;

    _PGFT_GetTextSize_INTERNAL(ft, font, font_size, render, font_text);

    width = PGFT_TRUNC(PGFT_ROUND(font_text->text_size.x));
    height = PGFT_TRUNC(PGFT_ROUND(font_text->text_size.y));

    /* if bg color exists, paint background */
    if (bgcolor)
    {
        SDL_Rect    bg_fill; 
        FT_UInt32   fillcolor;

        fillcolor = SDL_MapRGBA(surface->format, 
                bgcolor->r, bgcolor->g, bgcolor->b, 255);

        bg_fill.x = (FT_Int16)x;
        bg_fill.y = (FT_Int16)y;
        bg_fill.w = (FT_UInt16)width;
        bg_fill.h = (FT_UInt16)height;

        SDL_FillRect(surface, &bg_fill, fillcolor);
    }

    /*
     * Setup target surface struct
     */
    font_surf.buffer = surface->pixels;
    font_surf.buffer_cap = ((FT_Byte *)surface->pixels) + (surface->pitch * surface->h);
    font_surf.x_offset = x;
    font_surf.y_offset = y;

    font_surf.width = surface->w;
    font_surf.height = surface->h;
    font_surf.pitch = surface->pitch / surface->format->BytesPerPixel;

    font_surf.format = surface->format;

    if (render->antialias == FONT_RENDER_ANTIALIAS)
    {
        font_surf.render = __SDLrenderFuncs[surface->format->BytesPerPixel];
    }
    else
    {
        font_surf.render = __MONOrenderFuncs[surface->format->BytesPerPixel];
    }

    /*
     * Render!
     */
    if (_PGFT_Render_INTERNAL(ft, font, font_text, font_size, fgcolor, &font_surf, render) != 0)
        return -1;

    *_width = width;
    *_height = height;

    if (locked)
        SDL_UnlockSurface(surface);

    return 0;
}

PyObject *PGFT_Render_NewSurface(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    int ptsize, FontRenderMode *render, PyObject *text,
    PyColor *fgcolor, PyColor *bgcolor, int *_width, int *_height)
{
    int locked = 0;
    FT_UInt32 fillcolor, rmask, gmask, bmask, amask;
    SDL_Surface *surface = NULL;

    FontSurface font_surf;
    FontText *font_text;
    int width, height;

    /* build font text */
    font_text = PGFT_LoadFontText(ft, font, ptsize, render, text);

    if (!font_text)
        return NULL;

    if (_PGFT_GetTextSize_INTERNAL(ft, font, ptsize, render, font_text) != 0)
        return NULL;

    width = PGFT_TRUNC(PGFT_ROUND(font_text->text_size.x));
    height = PGFT_TRUNC(PGFT_ROUND(font_text->text_size.y));

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    rmask = 0xff000000;
    gmask = 0x00ff0000;
    bmask = 0x0000ff00;
    amask = 0x000000ff;
#else
    rmask = 0x000000ff;
    gmask = 0x0000ff00;
    bmask = 0x00ff0000;
    amask = 0xff000000;
#endif

    surface = SDL_CreateRGBSurface(SDL_SWSURFACE, 
            width, height,
            32, rmask, gmask, bmask, amask);

    if (!surface)
    {
        _PGFT_SetError(ft, SDL_GetError (), 0);
        return NULL;
    }

    if (SDL_MUSTLOCK(surface))
    {
        if (SDL_LockSurface(surface) == -1)
        {
            _PGFT_SetError(ft, SDL_GetError (), 0);
            SDL_FreeSurface(surface);
            return NULL;
        }
        locked = 1;
    }

    if (bgcolor)
    {
        fillcolor = SDL_MapRGBA(surface->format, 
                bgcolor->r, bgcolor->g, bgcolor->b, 255);
    }
    else
    {
        fillcolor = SDL_MapRGBA(surface->format, 0, 0, 0, 0);
    }

    SDL_FillRect(surface, NULL, fillcolor);

    font_surf.buffer = surface->pixels;
    font_surf.buffer_cap = ((FT_Byte *)surface->pixels) + (surface->pitch * surface->h);
    font_surf.x_offset = font_surf.y_offset = 0;

    font_surf.width = surface->w;
    font_surf.height = surface->h;
    font_surf.pitch = surface->pitch / sizeof(FT_UInt32);

    font_surf.format = surface->format;
    font_surf.render = __render_glyph_RGB4;

    if (_PGFT_Render_INTERNAL(ft, font, font_text, ptsize, fgcolor, &font_surf, render) != 0)
    {
        SDL_FreeSurface(surface);
        return NULL;
    }

    *_width = width;
    *_height = height;

    if (locked)
        SDL_UnlockSurface(surface);

    return PySDLSurface_NewFromSDLSurface(surface);
}
#endif




/*********************************************************
 *
 * Rendering on generic arrays
 *
 *********************************************************/
void __render_glyph_ByteArray(int x, int y, FontSurface *surface,
    FT_Bitmap *bitmap, PyColor *fg_color)
{
    FT_Byte *dst = ((FT_Byte *)surface->buffer) + x + (y * surface->pitch);
    FT_Byte *dst_cpy;

    const FT_Byte *src = bitmap->buffer;
    const FT_Byte *src_cpy;

    int j, i;

    for (j = 0; j < bitmap->rows; ++j)
    {
        src_cpy = src;
        dst_cpy = dst;

        LOOP_UNROLLED4({ *dst_cpy++ = (FT_Byte)(~(*src_cpy++)); }, i, bitmap->width);

        dst += surface->pitch;
        src += bitmap->pitch;
    }
}

PyObject *PGFT_Render_PixelArray(FreeTypeInstance *ft, PyFreeTypeFont *font,
    int ptsize, PyObject *text, int *_width, int *_height)
{
    FT_Byte *buffer = NULL;
    PyObject *array = NULL;
    FontSurface surf;
    int width, height;

    FontText *font_text;
    FontRenderMode render;
    int array_size;

    /* TODO: pixel arrays must also take custom rendering styles */
    PGFT_BuildRenderMode(&render, FT_STYLE_NORMAL,
            FONT_RENDER_HORIZONTAL, FONT_RENDER_ANTIALIAS, 0);

    /* build font text */
    font_text = PGFT_LoadFontText(ft, font, ptsize, &render, text);

    if (!font_text)
        goto cleanup;

    if (_PGFT_GetTextSize_INTERNAL(ft, font, ptsize, &render, font_text) != 0)
        goto cleanup;

    width = PGFT_TRUNC(PGFT_ROUND(font_text->text_size.x));
    height = PGFT_TRUNC(PGFT_ROUND(font_text->text_size.y));
    array_size = width * height;

    buffer = malloc((size_t)array_size);
    if (!buffer)
    {
        _PGFT_SetError(ft, "Could not allocate memory", 0);
        goto cleanup;
    }

    memset(buffer, 0xFF, (size_t)array_size);

    surf.buffer = buffer;
    surf.width = surf.pitch = width;
    surf.height = height;

    surf.format = NULL;
    surf.render = __render_glyph_ByteArray;

    if (_PGFT_Render_INTERNAL(ft, font, font_text, ptsize, 0x0, &surf, &render) != 0)
        goto cleanup;

    *_width = width;
    *_height = height;

    array = Bytes_FromStringAndSize((char *)buffer, array_size);

cleanup:
    if (buffer)
        free(buffer);

    return array;
}




/*********************************************************
 *
 * New rendering algorithm (rotation + veritical drawing)
 *
 *********************************************************/
int _PGFT_Render_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    FontText *text, int font_size, PyColor *fg_color, FontSurface *surface, 
    FontRenderMode *render)
{
    const FT_Fixed center = (1 << 15); // 0.5

    int         n;
    FT_Vector   pen, advances[MAX_GLYPHS];
    FT_Face     face;
    FT_Error    error;

    FT_Fixed    bold_str;
    FT_Matrix   shear_matrix;

    int         x = (surface->x_offset << 6);
    int         y = (surface->y_offset << 6);

    assert(text->text_size.x);
    assert(text->text_size.y);

    x += (text->text_size.x / 2);
    y += (text->text_size.y / 2);
    y -= (text->baseline_offset.y);

    /******************************************************
     * Load scaler, size & face
     ******************************************************/
    face = _PGFT_GetFaceSized(ft, font, font_size);

    if (!face)
    {
        _PGFT_SetError(ft, "Failed to resize face", 0);
        return -1;
    }

    /******************************************************
     * Load advance information
     ******************************************************/

    error = PGFT_GetTextAdvances(ft, font, font_size, render, text, advances);

    if (error)
    {
        _PGFT_SetError(ft, "Failed to load advance glyph advances", error);
        return error;
    }

    /******************************************************
     * Prepare pen for drawing 
     ******************************************************/

    /* change to Cartesian coordinates */
    y = (surface->height << 6) - y;

    /* get the extent, which we store in the last slot */
    pen = advances[text->length - 1];

    pen.x = FT_MulFix(pen.x, center); 
    pen.y = FT_MulFix(pen.y, center);

    if (!render->vertical)
        pen.y += PGFT_ROUND(text->glyph_size.y / 2);

    /* get pen position */
    if (render->matrix && FT_IS_SCALABLE(face))
    {
        FT_Vector_Transform(&pen, render->matrix);
        pen.x = x - pen.x;
        pen.y = y - pen.y;
    }
    else
    {
        pen.x = PGFT_ROUND(x - pen.x);
        pen.y = PGFT_ROUND(y - pen.y);
    }

    /******************************************************
     * Prepare data for glyph transformations
     ******************************************************/

    if (render->style & FT_STYLE_ITALIC)
    {
        shear_matrix.xx = (1<< 16);
        shear_matrix.xy = (FT_Fixed)(SLANT_FACTOR * (1 << 16));
        shear_matrix.yx = 0;
        shear_matrix.yy = (1 << 16);
    }

    if (render->style & FT_STYLE_BOLD)
    {
        bold_str = FT_MulFix(face->units_per_EM, face->size->metrics.y_scale);
        bold_str = (FT_Fixed)(bold_str * BOLD_FACTOR);
    }

    /******************************************************
     * Draw text
     ******************************************************/

    for (n = 0; n < text->length; ++n)
    {
        FT_Glyph image;
        FT_BBox bbox;

        FontGlyph *glyph = &(text->glyphs[n]);

        if (!glyph->image)
            continue;

        /* copy image */
        error = FT_Glyph_Copy(glyph->image, &image);
        if (error)
            continue;

        if (image->format == FT_GLYPH_FORMAT_OUTLINE)
        {
            if (render->style & FT_STYLE_BOLD)
            {
                FT_OutlineGlyph outline;
                outline = (FT_OutlineGlyph)image;
                FT_Outline_Embolden(&(outline->outline), bold_str);
            }

            if (render->style & FT_STYLE_ITALIC)
            {
                FT_OutlineGlyph outline;
                outline = (FT_OutlineGlyph)image;
                FT_Outline_Transform(&(outline->outline), &shear_matrix);
            }

            if (render->vertical)
                error = FT_Glyph_Transform(image, NULL, &glyph->vvector);

            if (!error)
                error = FT_Glyph_Transform(image, render->matrix, &pen);

            if (error)
            {
                FT_Done_Glyph(image);
                continue;
            }
        }
        else
        {
            FT_BitmapGlyph  bitmap = (FT_BitmapGlyph)image;

            if (render->vertical)
            {
                bitmap->left += (glyph->vvector.x + pen.x) >> 6;
                bitmap->top  += (glyph->vvector.x + pen.y) >> 6;
            }
            else
            {
                bitmap->left += pen.x >> 6;
                bitmap->top  += pen.y >> 6;
            }
        }

        if (render->matrix)
            FT_Vector_Transform(advances + n, render->matrix);

        pen.x += advances[n].x;
        pen.y += advances[n].y;

        FT_Glyph_Get_CBox(image, FT_GLYPH_BBOX_PIXELS, &bbox);

        if (bbox.xMax > 0 && bbox.yMax > 0 &&
            bbox.xMin < surface->width &&
            bbox.yMin < surface->height)
        {
            int         left, top;
            FT_Bitmap*  source;
            FT_BitmapGlyph  bitmap;

            if (image->format == FT_GLYPH_FORMAT_OUTLINE)
            {
                FT_Render_Mode render_mode = 
                    render->antialias ?
                    FT_RENDER_MODE_NORMAL :
                    FT_RENDER_MODE_MONO;

                /* render the glyph to a bitmap, don't destroy original */
                FT_Glyph_To_Bitmap(&image, render_mode, NULL, 0);
            }

            if (image->format != FT_GLYPH_FORMAT_BITMAP)
                continue;

            bitmap = (FT_BitmapGlyph)image;
            source = &bitmap->bitmap;

            left = bitmap->left;
            top = bitmap->top;

            top = surface->height - top;

            surface->render(left, top, surface, source, fg_color);
        }

        FT_Done_Glyph(image);
    } /* END OF RENDERING LOOP */

    return error;
}

