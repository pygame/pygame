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
#define NO_PYGAME_C_API

#include "ft_wrap.h"
#include FT_MODULE_H

#if defined (PGFT_PYGAME1_COMPAT)
#   include "ft_pixel.h"
#elif defined (HAVE_PYGAME_SDL_VIDEO)
#   include "surface.h"
#endif

void __render_glyph_GRAY1(int x, int y, FontSurface *surface,
                          const FT_Bitmap *bitmap, const FontColor *fg_color)
{
    FT_Byte *dst = ((FT_Byte *)surface->buffer) + x + (y * surface->pitch);
    FT_Byte *dst_cpy;

    const FT_Byte *src = bitmap->buffer;
    const FT_Byte *src_cpy;

    FT_Byte src_byte;
    int j, i;

    /*
     * Assumption, target buffer was filled with zeros before any rendering.
     */

    for (j = 0; j < bitmap->rows; ++j) {
        src_cpy = src;
        dst_cpy = dst;

        for (i = 0; i < bitmap->width; ++i) {
            src_byte = *src_cpy;
            if (src_byte) {
                *dst_cpy = src_byte + *dst_cpy - src_byte * *dst_cpy / 255;
            }
            ++src_cpy;
            ++dst_cpy;
        }

        dst += surface->pitch;
        src += bitmap->pitch;
    }
}

void __render_glyph_MONO_as_GRAY1(int x, int y, FontSurface *surface,
                                  const FT_Bitmap *bitmap,
                                  const FontColor *fg_color)
{
    const int off_x = (x < 0) ? -x : 0;
    const int off_y = (y < 0) ? -y : 0;

    const int max_x = MIN(x + bitmap->width, surface->width);
    const int max_y = MIN(y + bitmap->rows, surface->height);

    const int rx = MAX(0, x);
    const int ry = MAX(0, y);

    int i, j, shift;
    const unsigned char* src;
    unsigned char* dst;
    const unsigned char*  src_cpy;
    unsigned char* dst_cpy;
    FT_UInt32 val;
    FT_Byte shade = fg_color->a;

    src  = bitmap->buffer + (off_y * bitmap->pitch) + (off_x >> 3);
    dst = (unsigned char *)surface->buffer + rx + (ry * surface->pitch);

    shift = off_x & 7;

    for (j = ry; j < max_y; ++j) {
        src_cpy = src;
        dst_cpy = dst;
        val = (FT_UInt32)(*src_cpy++ | 0x100) << shift;

        for (i = rx; i < max_x; ++i, ++dst_cpy) {
            if (val & 0x10000) {
                val = (FT_UInt32)(*src_cpy++ | 0x100);
            }

            if (val & 0x80) {
                *dst_cpy = shade;
            }

            val   <<= 1;
        }

        src += bitmap->pitch;
        dst += surface->pitch;
    }
}

void __render_glyph_GRAY_as_MONO1(int x, int y, FontSurface *surface,
                                  const FT_Bitmap *bitmap,
                                  const FontColor *fg_color)
{
    FT_Byte *dst = ((FT_Byte *)surface->buffer) + x + (y * surface->pitch);
    FT_Byte *dst_cpy;
    FT_Byte shade = fg_color->a;

    const FT_Byte *src = bitmap->buffer;
    const FT_Byte *src_cpy;

    int j, i;

    /*
     * Assumption, target buffer was filled with the background color before
     * any rendering.
     */

    for (j = 0; j < bitmap->rows; ++j) {
        src_cpy = src;
        dst_cpy = dst;

        for (i = 0; i < bitmap->width; ++i) {
            if (*src_cpy & '\200') /* Round up on 128 */ {
                *dst_cpy = shade;
            }
            ++src_cpy;
            ++dst_cpy;
        }

        dst += surface->pitch;
        src += bitmap->pitch;
    }
}

void __fill_glyph_GRAY1(FT_Fixed x, FT_Fixed y, FT_Fixed w, FT_Fixed h,
                        FontSurface *surface, const FontColor *color)
{
    int i, j;
    FT_Byte *dst;
    FT_Byte *dst_cpy;
    FT_Byte shade = color->a;
    FT_Byte edge_shade;

    x = MAX(0, x);
    y = MAX(0, y);

    if (x + w > INT_TO_FX6(surface->width)) {
        w = INT_TO_FX6(surface->width) - x;
    }
    if (y + h > INT_TO_FX6(surface->height)) {
        h = INT_TO_FX6(surface->height) - y;
    }

    dst = ((FT_Byte *)surface->buffer +
           FX6_TRUNC(FX6_CEIL(x)) +
           FX6_TRUNC(FX6_CEIL(y)) * surface->pitch);

    if (y < FX6_CEIL(y)) {
        dst_cpy = dst - surface->pitch;
        edge_shade = FX6_TRUNC(FX6_ROUND(shade * (FX6_CEIL(y) - y)));

        for (i = 0; i < FX6_TRUNC(FX6_CEIL(w)); ++i, ++dst_cpy) {
            *dst_cpy = edge_shade;
        }
    }

    for (j = 0; j < FX6_TRUNC(FX6_FLOOR(h + y) - FX6_CEIL(y)); ++j) {
        dst_cpy = dst;

        for (i = 0; i < FX6_TRUNC(FX6_CEIL(w)); ++i, ++dst_cpy) {
            *dst_cpy = shade;
        }

        dst += surface->pitch;
    }

    if (h > FX6_FLOOR(h + y) - y) {
        dst_cpy = dst;
        edge_shade = FX6_TRUNC(FX6_ROUND(shade * (y + y - FX6_FLOOR(h + y))));
        for (i = 0; i < FX6_TRUNC(FX6_CEIL(w)); ++i, ++dst_cpy) {
            *dst_cpy = edge_shade;
        }
    }
}

void __render_glyph_INT(int x, int y, FontSurface *surface,
                        const FT_Bitmap *bitmap, const FontColor *fg_color)
{
    FT_Byte *dst = ((FT_Byte *)surface->buffer +
                    x * surface->item_stride + y * surface->pitch);
    int item_size = surface->format->BytesPerPixel;
    int item_stride = surface->item_stride;
    FT_Byte *dst_cpy;

    const FT_Byte *src = bitmap->buffer;
    const FT_Byte *src_cpy;
    FT_Byte src_byte;
    FT_Byte dst_byte;
    FT_Byte mask = ~fg_color->a;

    int j, i;
    int b, int_offset;

    /*
     * Assumption, target buffer was filled with the background color before
     * any rendering.
     */

    if (item_size == 1) {
        for (j = 0; j < bitmap->rows; ++j) {
            src_cpy = src;
            dst_cpy = dst;

            for (i = 0; i < bitmap->width; ++i) {
                src_byte = *src_cpy;
                if (src_byte) {
                    *dst_cpy = ((src_byte + *dst_cpy -
                                src_byte * *dst_cpy / 255) ^ mask);
                }
                ++src_cpy;
                dst_cpy += item_stride;
            }

            dst += surface->pitch;
            src += bitmap->pitch;
        }
    }
    else {
        int_offset = surface->format->Ashift / 8;

        for (j = 0; j < bitmap->rows; ++j) {
            src_cpy = src;
            dst_cpy = dst;

            for (i = 0; i < bitmap->width; ++i) {
                dst_byte = dst_cpy[int_offset];
                for (b = 0; b < item_size; ++b) {
                    dst_cpy[b] = 0;
                }

                src_byte = *src_cpy;
                if (src_byte) {
                    dst_cpy[int_offset] = ((src_byte + dst_byte -
                                            src_byte * dst_byte / 255) ^ mask);
                }
                ++src_cpy;
                dst_cpy += item_stride;
            }

            dst += surface->pitch;
            src += bitmap->pitch;
        }
    }
}

void __render_glyph_MONO_as_INT(int x, int y, FontSurface *surface,
                                const FT_Bitmap *bitmap,
                                const FontColor *fg_color)
{
    const int off_x = (x < 0) ? -x : 0;
    const int off_y = (y < 0) ? -y : 0;

    const int max_x = MIN(x + bitmap->width, surface->width);
    const int max_y = MIN(y + bitmap->rows, surface->height);

    const int rx = MAX(0, x);
    const int ry = MAX(0, y);

    int i, j, shift;
    int b, int_offset;
    int item_stride = surface->item_stride;
    int item_size = surface->format->BytesPerPixel;
    unsigned char *src;
    unsigned char *dst;
    unsigned char *src_cpy;
    unsigned char *dst_cpy;
    FT_UInt32 val;
    FT_Byte shade = fg_color->a;

    /*
     * Assumption, target buffer was filled with the background color before
     * any rendering.
     */

    src  = bitmap->buffer + (off_y * bitmap->pitch) + (off_x >> 3);
    dst = ((unsigned char *)surface->buffer +
           rx * surface->item_stride + ry * surface->pitch);

    shift = off_x & 7;

    if (item_size == 1) {
        /* Slightly optimized loop for 1 byte target int */
        for (j = ry; j < max_y; ++j) {
            src_cpy = src;
            dst_cpy = dst;
            val = (FT_UInt32)(*src_cpy++ | 0x100) << shift;

            for (i = rx; i < max_x; ++i, dst_cpy += item_stride) {
                if (val & 0x10000) {
                    val = (FT_UInt32)(*src_cpy++ | 0x100);
                }

                if (val & 0x80) {
                    *dst_cpy = shade;
                }

                val   <<= 1;
            }

            src += bitmap->pitch;
            dst += surface->pitch;
        }
    }
    else {
        /* Generic copy for arbitrary target int size */
        int_offset = surface->format->Ashift / 8;

        for (j = ry; j < max_y; ++j) {
            src_cpy = src;
            dst_cpy = dst;
            val = (FT_UInt32)(*src_cpy++ | 0x100) << shift;

            for (i = rx; i < max_x; ++i, dst_cpy += item_stride) {
                for (b = 0; b < item_size; ++b) {
                    dst_cpy[b] = 0;
                }

                if (val & 0x10000) {
                    val = (FT_UInt32)(*src_cpy++ | 0x100);
                }

                if (val & 0x80) {
                    dst_cpy[int_offset] = shade;
                }

                val   <<= 1;
            }

            src += bitmap->pitch;
            dst += surface->pitch;
        }
    }
}

void __fill_glyph_INT(FT_Fixed x, FT_Fixed y, FT_Fixed w, FT_Fixed h,
                      FontSurface *surface, const FontColor *color)
{
    int b, i, j;
    FT_Byte *dst;
    int itemsize = surface->format->BytesPerPixel;
    int item_stride = surface->item_stride;
    int byteoffset = surface->format->Ashift / 8;
    FT_Byte *dst_cpy;
    FT_Byte shade = color->a;
    FT_Byte edge_shade;

    x = MAX(0, x);
    y = MAX(0, y);

    if (x + w > INT_TO_FX6(surface->width)) {
        w = INT_TO_FX6(surface->width) - x;
    }
    if (y + h > INT_TO_FX6(surface->height)) {
        h = INT_TO_FX6(surface->height) - y;
    }

    dst = ((FT_Byte *)surface->buffer +
           FX6_TRUNC(FX6_CEIL(x)) * itemsize +
           FX6_TRUNC(FX6_CEIL(y)) * surface->pitch);

    if (itemsize == 1) {
        if (y < FX6_CEIL(y)) {
            dst_cpy = dst - surface->pitch;
            edge_shade = FX6_TRUNC(FX6_ROUND(shade * (FX6_CEIL(y) - y)));

            for (i = 0;
                 i < FX6_TRUNC(FX6_CEIL(w));
                 ++i, dst_cpy += item_stride) {
                *dst_cpy = edge_shade;
            }
        }

        for (j = 0; j < FX6_TRUNC(FX6_FLOOR(h + y) - FX6_CEIL(y)); ++j) {
            dst_cpy = dst;

            for (i = 0;
                 i < FX6_TRUNC(FX6_CEIL(w));
                 ++i, dst_cpy += item_stride) {
                *dst_cpy = shade;
            }

            dst += surface->pitch;
        }

        if (h > FX6_FLOOR(h + y) - y) {
            dst_cpy = dst;
            edge_shade = FX6_TRUNC(FX6_ROUND(shade *
                                             (y + y - FX6_FLOOR(h + y))));
            for (i = 0;
                 i < FX6_TRUNC(FX6_CEIL(w));
                 ++i, dst_cpy += item_stride) {
                *dst_cpy = edge_shade;
            }
        }
    }
    else {
        if (y < FX6_CEIL(y)) {
            dst_cpy = dst - surface->pitch;
            edge_shade = FX6_TRUNC(FX6_ROUND(shade * (FX6_CEIL(y) - y)));

            for (i = 0;
                 i < FX6_TRUNC(FX6_CEIL(w));
                 ++i, dst_cpy += item_stride) {
                for (b = 0; b < itemsize; ++b) {
                    dst_cpy[b] = 0;
                }
                dst_cpy[byteoffset] = edge_shade;
            }
        }

        for (j = 0; j < FX6_TRUNC(FX6_FLOOR(h + y) - FX6_CEIL(y)); ++j) {
            dst_cpy = dst;

            for (i = 0;
                 i < FX6_TRUNC(FX6_CEIL(w));
                 ++i, dst_cpy += item_stride) {
                for (b = 0; b < itemsize; ++b) {
                    dst_cpy[b] = 0;
                }
                dst_cpy[byteoffset] = shade;
            }

            dst += surface->pitch;
        }

        if (h > FX6_FLOOR(h + y) - y) {
            dst_cpy = dst;
            edge_shade = FX6_TRUNC(FX6_ROUND(shade *
                                             (h + y - FX6_FLOOR(h + y))));
            for (i = 0;
                 i < FX6_TRUNC(FX6_CEIL(w));
                 ++i, dst_cpy += item_stride) {
                for (b = 0; b < itemsize; ++b) {
                    dst_cpy[b] = 0;
                }
                dst_cpy[byteoffset] = edge_shade;
            }
        }
    }
}

#ifdef HAVE_PYGAME_SDL_VIDEO

#define _CREATE_RGB_FILLER(_bpp, _getp, _setp, _blendp)     \
    void __fill_glyph_RGB##_bpp(FT_Fixed x, FT_Fixed y,     \
                                FT_Fixed w, FT_Fixed h,     \
                                FontSurface *surface,       \
                                const FontColor *color)     \
    {                                                       \
        int i, j;                                           \
        unsigned char *dst;                                 \
        FT_UInt32 bgR, bgG, bgB, bgA;                       \
        FT_Byte edge_a;                                     \
                                                            \
        x = MAX(0, x);                                      \
        y = MAX(0, y);                                      \
                                                            \
        if (x + w > INT_TO_FX6(surface->width)) {           \
            w = INT_TO_FX6(surface->width) - x;             \
        }                                                   \
        if (y + h > INT_TO_FX6(surface->height)) {          \
            h = INT_TO_FX6(surface->height) - y;            \
        }                                                   \
                                                            \
        dst = ((FT_Byte *)surface->buffer +                 \
               FX6_TRUNC(FX6_CEIL(x)) * _bpp +              \
               FX6_TRUNC(FX6_CEIL(y)) * surface->pitch);    \
                                                            \
        if (y < FX6_CEIL(y)) {                              \
            unsigned char *_dst = dst - surface->pitch;     \
                                                            \
            edge_a = (                                      \
                FX6_TRUNC(FX6_ROUND(color->a *              \
                                    (FX6_CEIL(y) - y))));   \
                                                            \
            for (i = 0;                                     \
                 i < FX6_TRUNC(FX6_CEIL(w));                \
                 ++i, _dst += _bpp) {                       \
                FT_UInt32 pixel = (FT_UInt32)_getp;         \
                                                            \
                if (_bpp == 1) {                            \
                    GET_PALETTE_VALS(                       \
                            pixel, surface->format,         \
                            bgR, bgG, bgB, bgA);            \
                }                                           \
                else {                                      \
                    GET_RGB_VALS(                           \
                            pixel, surface->format,         \
                            bgR, bgG, bgB, bgA);            \
                                                            \
                }                                           \
                                                            \
                ALPHA_BLEND(                                \
                        color->r, color->g, color->b,       \
                        edge_a, bgR, bgG, bgB, bgA);        \
                                                            \
                _blendp;                                    \
            }                                               \
        }                                                   \
                                                            \
        for (j = 0;                                         \
             j < FX6_TRUNC(FX6_FLOOR(h + y) - FX6_CEIL(y)); \
             ++j) {                                         \
            unsigned char *_dst = dst;                      \
                                                            \
            for (i = 0;                                     \
                 i < FX6_TRUNC(FX6_CEIL(w));                \
                 ++i, _dst += _bpp) {                       \
                FT_UInt32 pixel = (FT_UInt32)_getp;         \
                                                            \
                if (_bpp == 1) {                            \
                    GET_PALETTE_VALS(                       \
                            pixel, surface->format,         \
                            bgR, bgG, bgB, bgA);            \
                }                                           \
                else {                                      \
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
            dst += surface->pitch;                          \
        }                                                   \
                                                            \
        if (h > FX6_FLOOR(h + y) - y) {                     \
            unsigned char *_dst = dst;                      \
                                                            \
            edge_a = (                                      \
                FX6_TRUNC(FX6_ROUND(color->a *              \
                                    (h + y -                \
                                     FX6_FLOOR(h + y)))));  \
                                                            \
            for (i = 0;                                     \
                 i < FX6_TRUNC(FX6_CEIL(w));                \
                 ++i, _dst += _bpp) {                       \
                FT_UInt32 pixel = (FT_UInt32)_getp;         \
                                                            \
                if (_bpp == 1) {                            \
                    GET_PALETTE_VALS(                       \
                            pixel, surface->format,         \
                            bgR, bgG, bgB, bgA);            \
                }                                           \
                else {                                      \
                    GET_RGB_VALS(                           \
                            pixel, surface->format,         \
                            bgR, bgG, bgB, bgA);            \
                                                            \
                }                                           \
                                                            \
                ALPHA_BLEND(                                \
                        color->r, color->g, color->b,       \
                        edge_a, bgR, bgG, bgB, bgA);        \
                                                            \
                _blendp;                                    \
            }                                               \
        }                                                   \
    }

#define __MONO_RENDER_INNER_LOOP(_bpp, _code)               \
    for (j = ry; j < max_y; ++j)                            \
    {                                                       \
        const unsigned char* _src = src;                    \
        unsigned char* _dst = dst;                          \
        FT_UInt32 val =                                     \
            (FT_UInt32)(*_src++ | 0x100) << shift;          \
                                                            \
        for (i = rx; i < max_x; ++i, _dst += _bpp) {        \
            if (val & 0x10000) {                            \
                val = (FT_UInt32)(*_src++ | 0x100);         \
            }                                               \
            if (val & 0x80) {                               \
                _code;                                      \
            }                                               \
            val <<= 1;                                      \
        }                                                   \
                                                            \
        src += bitmap->pitch;                               \
        dst += surface->pitch;                              \
    }                                                       \

#define _CREATE_MONO_RENDER(_bpp, _getp, _setp, _blendp)    \
    void __render_glyph_MONO##_bpp(int x, int y,            \
                                   FontSurface *surface,    \
                                   const FT_Bitmap *bitmap, \
                                   const FontColor *color)  \
    {                                                       \
        const int off_x = (x < 0) ? -x : 0;                 \
        const int off_y = (y < 0) ? -y : 0;                 \
                                                            \
        const int max_x =                                   \
             MIN(x + bitmap->width, surface->width);        \
        const int max_y =                                   \
             MIN(y + bitmap->rows, surface->height);        \
                                                            \
        const int rx = MAX(0, x);                           \
        const int ry = MAX(0, y);                           \
                                                            \
        int i, j, shift;                                                \
        const unsigned char* src;                                       \
        unsigned char* dst;                                             \
        FT_UInt32 full_color;                                           \
        FT_UInt32 bgR, bgG, bgB, bgA;                                   \
                                                                        \
        src  = bitmap->buffer + (off_y * bitmap->pitch) + (off_x >> 3); \
        dst = (unsigned char *)surface->buffer + (rx * _bpp) +          \
                    (ry * surface->pitch);                              \
                                                                        \
        full_color = SDL_MapRGBA(surface->format, (FT_Byte)color->r,    \
                (FT_Byte)color->g, (FT_Byte)color->b, 255);             \
                                                                        \
        shift = off_x & 7;                                              \
                                                                        \
        if (color->a == 0xFF) {                                         \
            __MONO_RENDER_INNER_LOOP(_bpp,                              \
            {                                                           \
                _setp;                                                  \
            });                                                         \
        }                                                               \
        else if (color->a > 0) {                                        \
            __MONO_RENDER_INNER_LOOP(_bpp,                              \
            {                                                           \
                FT_UInt32 pixel = (FT_UInt32)_getp;                     \
                                                                        \
                if (_bpp == 1) {                                        \
                    GET_PALETTE_VALS(                                   \
                            pixel, surface->format,                     \
                            bgR, bgG, bgB, bgA);                        \
                }                                                       \
                else {                                                  \
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
                                  const FT_Bitmap *bitmap,              \
                                  const FontColor *color)               \
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
                        (ry * surface->pitch);                          \
        FT_Byte *_dst;                                                  \
                                                                        \
        const FT_Byte *src = bitmap->buffer + off_x +                   \
                                (off_y * bitmap->pitch);                \
        const FT_Byte *_src;                                            \
                                                                        \
        _DECLARE_full_color##_bpp(surface, color)                       \
        /*                                                              \
        const FT_UInt32 full_color =                                    \
            SDL_MapRGBA(surface->format, (FT_Byte)color->r,             \
                    (FT_Byte)color->g, (FT_Byte)color->b, 255);         \
        */                                                              \
                                                                        \
        FT_UInt32 bgR, bgG, bgB, bgA;                                   \
        int j, i;                                                       \
                                                                        \
        for (j = ry; j < max_y; ++j) {                                  \
            _src = src;                                                 \
            _dst = dst;                                                 \
                                                                        \
            for (i = rx; i < max_x; ++i, _dst += _bpp) {                \
                FT_UInt32 alpha = (*_src++);                            \
                alpha = (alpha * color->a) / 255;                       \
                                                                        \
                if (alpha == 0xFF) {                                    \
                    _setp;                                              \
                }                                                       \
                else if (alpha > 0) {                                   \
                    FT_UInt32 pixel = (FT_UInt32)_getp;                 \
                                                                        \
                if (_bpp == 1) {                                        \
                    GET_PALETTE_VALS(                                   \
                            pixel, surface->format,                     \
                            bgR, bgG, bgB, bgA);                        \
                }                                                       \
                else {                                                  \
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
            dst += surface->pitch;                                      \
            src += bitmap->pitch;                                       \
        }                                                               \
    }

/* These macros removes a gcc unused variable warning for __render_glyph_RGB3 */
#define _DECLARE_full_color(s, c) const FT_UInt32 full_color =          \
    SDL_MapRGBA((s)->format, (FT_Byte)(c)->r, (FT_Byte)(c)->g,          \
                (FT_Byte)(c)->b, 255);
#define _DECLARE_full_color1(s, c) _DECLARE_full_color(s, c)
#define _DECLARE_full_color2(s, c) _DECLARE_full_color(s, c)
#define _DECLARE_full_color3(s, c)
#define _DECLARE_full_color4(s, c) _DECLARE_full_color(s, c)


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
    ((bgA >> surface->format->Aloss) << surface->format->Ashift  &  \
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
