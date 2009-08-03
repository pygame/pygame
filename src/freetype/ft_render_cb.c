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

#if defined (PGFT_PYGAME1_COMPAT)
#   include "ft_pixel.h"
#elif defined (HAVE_PYGAME_SDL_VIDEO)
#   include "surface.h"
#endif

void __render_glyph_ByteArray(int x, int y, FontSurface *surface,
    FT_Bitmap *bitmap, FontColor *fg_color)
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

        for (i = 0; i < bitmap->width; ++i)
            *dst_cpy++ = (FT_Byte)(~(*src_cpy++));

        dst += surface->pitch;
        src += bitmap->pitch;
    }
}

#ifdef HAVE_PYGAME_SDL_VIDEO

#define _CREATE_RGB_FILLER(_bpp, _getp, _setp, _blendp)     \
    void __fill_glyph_RGB##_bpp(int x, int y, int w, int h, \
            FontSurface *surface, FontColor *color)           \
    {                                                       \
        int i, j;                                           \
        unsigned char *dst;                                 \
        FT_UInt32 bgR, bgG, bgB, bgA;                       \
                                                            \
        x = MAX(0, x);                                      \
        y = MAX(0, y);                                      \
                                                            \
        if (x + w > surface->width)                         \
            w = surface->width - x;                         \
                                                            \
        if (y + h > surface->height)                        \
            h = surface->height - y;                        \
                                                            \
        dst = (unsigned char *)surface->buffer + (x * _bpp) +\
              (y * _bpp * surface->pitch);                  \
                                                            \
        for (j = 0; j < h; ++j)                             \
        {                                                   \
            unsigned char *_dst = dst;                      \
                                                            \
            for (i = 0; i < w; ++i, _dst += _bpp)           \
            {                                               \
                FT_UInt32 pixel = (FT_UInt32)_getp;         \
                                                            \
                if (_bpp == 1)                              \
                {                                           \
                    GET_PALETTE_VALS(                       \
                            pixel, surface->format,         \
                            bgR, bgG, bgB, bgA);            \
                }                                           \
                else                                        \
                {                                           \
                    GET_RGB_VALS(                           \
                            pixel, surface->format,         \
                            bgR, bgG, bgB, bgA);            \
                                                            \
                }                                           \
                                                            \
                ALPHA_BLEND(                                \
                        color->r, color->g, color->b,       \
                        color->a, bgR, bgG, bgB, bgA);      \
                                                            \
                _blendp;                                    \
            }                                               \
                                                            \
            dst += surface->pitch * _bpp;                   \
        }                                                   \
    }

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
            FT_Bitmap *bitmap, FontColor *color)                          \
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
                if (_bpp == 1)                                          \
                {                                                       \
                    GET_PALETTE_VALS(                                   \
                            pixel, surface->format,                     \
                            bgR, bgG, bgB, bgA);                        \
                }                                                       \
                else                                                    \
                {                                                       \
                    GET_RGB_VALS(                                       \
                            pixel, surface->format,                     \
                            bgR, bgG, bgB, bgA);                        \
                                                                        \
                }                                                       \
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
        FT_Bitmap *bitmap, FontColor *color)                            \
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
                if (_bpp == 1)                                          \
                {                                                       \
                    GET_PALETTE_VALS(                                   \
                            pixel, surface->format,                     \
                            bgR, bgG, bgB, bgA);                        \
                }                                                       \
                else                                                    \
                {                                                       \
                    GET_RGB_VALS(                                       \
                            pixel, surface->format,                     \
                            bgR, bgG, bgB, bgA);                        \
                                                                        \
                }                                                       \
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

_CREATE_RGB_FILLER(4,  _GET_PIXEL(FT_UInt32),   _SET_PIXEL(FT_UInt32),  _BLEND_PIXEL(FT_UInt32))
_CREATE_RGB_FILLER(3,  GET_PIXEL24(_dst),       _SET_PIXEL_24,          _BLEND_PIXEL_24)
_CREATE_RGB_FILLER(2,  _GET_PIXEL(FT_UInt16),   _SET_PIXEL(FT_UInt16),  _BLEND_PIXEL(FT_UInt16))
_CREATE_RGB_FILLER(1,  _GET_PIXEL(FT_Byte),     _SET_PIXEL(FT_Byte),    _BLEND_PIXEL_GENERIC(FT_Byte))
#endif
