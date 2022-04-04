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
#include "ft_pixel.h"

void
__render_glyph_GRAY1(int x, int y, FontSurface *surface,
                     const FT_Bitmap *bitmap, const FontColor *fg_color)
{
    FT_Byte *dst = ((FT_Byte *)surface->buffer) + x + (y * surface->pitch);
    FT_Byte *dst_cpy;

    const FT_Byte *src = bitmap->buffer;
    const FT_Byte *src_cpy;

    FT_Byte src_byte;
    unsigned int j, i;

#ifndef NDEBUG
    const FT_Byte *src_end = src + (bitmap->rows * bitmap->pitch);
    const FT_Byte *dst_end =
        ((FT_Byte *)surface->buffer + (surface->height * surface->pitch));
#endif

    assert(dst >= (FT_Byte *)surface->buffer && dst < dst_end);

    /*
     * Assumption, target buffer was filled with zeros before any rendering.
     */

    for (j = 0; j < (unsigned int)bitmap->rows; ++j) {
        src_cpy = src;
        dst_cpy = dst;

        for (i = 0; i < (unsigned int)bitmap->width; ++i) {
            assert(src_cpy < src_end);
            src_byte = *src_cpy;
            if (src_byte) {
                assert(dst_cpy < dst_end);
                *dst_cpy = src_byte + *dst_cpy - src_byte * *dst_cpy / 255;
            }
            ++src_cpy;
            ++dst_cpy;
        }

        dst += surface->pitch;
        src += bitmap->pitch;
    }
}

void
__render_glyph_MONO_as_GRAY1(int x, int y, FontSurface *surface,
                             const FT_Bitmap *bitmap,
                             const FontColor *fg_color)
{
    const int off_x = (x < 0) ? -x : 0;
    const int off_y = (y < 0) ? -y : 0;

    const int max_x = MIN(x + (int)bitmap->width, (int)surface->width);
    const int max_y = MIN(y + (int)bitmap->rows, (int)surface->height);

    const int rx = MAX(0, x);
    const int ry = MAX(0, y);

    int i, j, shift;
    const unsigned char *src;
    unsigned char *dst;
    const unsigned char *src_cpy;
    unsigned char *dst_cpy;
    FT_UInt32 val;
    FT_Byte shade = fg_color->a;

    src = bitmap->buffer + (off_y * bitmap->pitch) + (off_x >> 3);
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

            val <<= 1;
        }

        src += bitmap->pitch;
        dst += surface->pitch;
    }
}

void
__render_glyph_GRAY_as_MONO1(int x, int y, FontSurface *surface,
                             const FT_Bitmap *bitmap,
                             const FontColor *fg_color)
{
    FT_Byte *dst = ((FT_Byte *)surface->buffer) + x + (y * surface->pitch);
    FT_Byte *dst_cpy;
    FT_Byte shade = fg_color->a;

    const FT_Byte *src = bitmap->buffer;
    const FT_Byte *src_cpy;

    unsigned int j, i;

    /*
     * Assumption, target buffer was filled with the background color before
     * any rendering.
     */

    for (j = 0; j < (unsigned int)bitmap->rows; ++j) {
        src_cpy = src;
        dst_cpy = dst;

        for (i = 0; i < (unsigned int)bitmap->width; ++i) {
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

void
__fill_glyph_GRAY1(FT_Fixed x, FT_Fixed y, FT_Fixed w, FT_Fixed h,
                   FontSurface *surface, const FontColor *color)
{
    int i, j;
    FT_Byte *dst;
    FT_Byte *dst_cpy;
    FT_Byte shade = color->a;
    FT_Byte edge_shade;

#ifndef NDEBUG
    FT_Byte *dst_end =
        ((FT_Byte *)surface->buffer + (surface->height * surface->pitch));
#endif

    x = MAX(0, x);
    y = MAX(0, y);

    if (x + w > INT_TO_FX6(surface->width)) {
        w = INT_TO_FX6(surface->width) - x;
    }
    if (y + h > INT_TO_FX6(surface->height)) {
        h = INT_TO_FX6(surface->height) - y;
    }

    dst = ((FT_Byte *)surface->buffer + FX6_TRUNC(FX6_CEIL(x)) +
           FX6_TRUNC(FX6_CEIL(y)) * surface->pitch);

    if (y < FX6_CEIL(y)) {
        dst_cpy = dst - surface->pitch;
        edge_shade = FX6_TRUNC(FX6_ROUND(shade * (FX6_CEIL(y) - y)));

        for (i = 0; i < FX6_TRUNC(FX6_CEIL(w)); ++i, ++dst_cpy) {
            assert(dst_cpy < dst_end);
            *dst_cpy = edge_shade;
        }
    }

    for (j = 0; j < FX6_TRUNC(FX6_FLOOR(h + y) - FX6_CEIL(y)); ++j) {
        dst_cpy = dst;

        for (i = 0; i < FX6_TRUNC(FX6_CEIL(w)); ++i, ++dst_cpy) {
            assert(dst_cpy < dst_end);
            *dst_cpy = shade;
        }

        dst += surface->pitch;
    }

    if (h > FX6_FLOOR(h + y) - y) {
        dst_cpy = dst;
        edge_shade = FX6_TRUNC(FX6_ROUND(shade * (y + y - FX6_FLOOR(h + y))));
        for (i = 0; i < FX6_TRUNC(FX6_CEIL(w)); ++i, ++dst_cpy) {
            assert(dst_cpy < dst_end);
            *dst_cpy = edge_shade;
        }
    }
}

void
__render_glyph_INT(int x, int y, FontSurface *surface, const FT_Bitmap *bitmap,
                   const FontColor *fg_color)
{
    FT_Byte *dst = ((FT_Byte *)surface->buffer + x * surface->item_stride +
                    y * surface->pitch);
    int item_size = surface->format->BytesPerPixel;
    int item_stride = surface->item_stride;
    FT_Byte *dst_cpy;

    const FT_Byte *src = bitmap->buffer;
    const FT_Byte *src_cpy;
    FT_Byte src_byte;
    FT_Byte mask = ~fg_color->a;

    unsigned int j, i;

    /*
     * Assumption, target buffer was filled with the background color before
     * any rendering.
     */

    if (item_size == 1) {
        for (j = 0; j < (unsigned int)bitmap->rows; ++j) {
            src_cpy = src;
            dst_cpy = dst;

            for (i = 0; i < (unsigned int)bitmap->width; ++i) {
                src_byte = *src_cpy;
                if (src_byte) {
                    *dst_cpy =
                        ((src_byte + *dst_cpy - src_byte * *dst_cpy / 255) ^
                         mask);
                }
                ++src_cpy;
                dst_cpy += item_stride;
            }

            dst += surface->pitch;
            src += bitmap->pitch;
        }
    }
    else {
        FT_Byte dst_byte;
        int b, int_offset = surface->format->Ashift / 8;

        for (j = 0; j < (unsigned int)bitmap->rows; ++j) {
            src_cpy = src;
            dst_cpy = dst;

            for (i = 0; i < (unsigned int)bitmap->width; ++i) {
                dst_byte = dst_cpy[int_offset];
                for (b = 0; b < item_size; ++b) {
                    dst_cpy[b] = 0;
                }

                src_byte = *src_cpy;
                if (src_byte) {
                    dst_cpy[int_offset] =
                        ((src_byte + dst_byte - src_byte * dst_byte / 255) ^
                         mask);
                }
                ++src_cpy;
                dst_cpy += item_stride;
            }

            dst += surface->pitch;
            src += bitmap->pitch;
        }
    }
}

void
__render_glyph_MONO_as_INT(int x, int y, FontSurface *surface,
                           const FT_Bitmap *bitmap, const FontColor *fg_color)
{
    const int off_x = (x < 0) ? -x : 0;
    const int off_y = (y < 0) ? -y : 0;

    const int max_x = MIN(x + (int)bitmap->width, (int)surface->width);
    const int max_y = MIN(y + (int)bitmap->rows, (int)surface->height);

    const int rx = MAX(0, x);
    const int ry = MAX(0, y);

    int i, j, shift;
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

    src = bitmap->buffer + (off_y * bitmap->pitch) + (off_x >> 3);
    dst = ((unsigned char *)surface->buffer + rx * surface->item_stride +
           ry * surface->pitch);

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

                val <<= 1;
            }

            src += bitmap->pitch;
            dst += surface->pitch;
        }
    }
    else {
        /* Generic copy for arbitrary target int size */
        int b, int_offset = surface->format->Ashift / 8;

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

                val <<= 1;
            }

            src += bitmap->pitch;
            dst += surface->pitch;
        }
    }
}

void
__fill_glyph_INT(FT_Fixed x, FT_Fixed y, FT_Fixed w, FT_Fixed h,
                 FontSurface *surface, const FontColor *color)
{
    int i, j;
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

    dst = ((FT_Byte *)surface->buffer + FX6_TRUNC(FX6_CEIL(x)) * itemsize +
           FX6_TRUNC(FX6_CEIL(y)) * surface->pitch);

    if (itemsize == 1) {
        if (y < FX6_CEIL(y)) {
            dst_cpy = dst - surface->pitch;
            edge_shade = FX6_TRUNC(FX6_ROUND(shade * (FX6_CEIL(y) - y)));

            for (i = 0; i < FX6_TRUNC(FX6_CEIL(w));
                 ++i, dst_cpy += item_stride) {
                *dst_cpy = edge_shade;
            }
        }

        for (j = 0; j < FX6_TRUNC(FX6_FLOOR(h + y) - FX6_CEIL(y)); ++j) {
            dst_cpy = dst;

            for (i = 0; i < FX6_TRUNC(FX6_CEIL(w));
                 ++i, dst_cpy += item_stride) {
                *dst_cpy = shade;
            }

            dst += surface->pitch;
        }

        if (h > FX6_FLOOR(h + y) - y) {
            dst_cpy = dst;
            edge_shade =
                FX6_TRUNC(FX6_ROUND(shade * (y + y - FX6_FLOOR(h + y))));
            for (i = 0; i < FX6_TRUNC(FX6_CEIL(w));
                 ++i, dst_cpy += item_stride) {
                *dst_cpy = edge_shade;
            }
        }
    }
    else {
        int b;

        if (y < FX6_CEIL(y)) {
            dst_cpy = dst - surface->pitch;
            edge_shade = FX6_TRUNC(FX6_ROUND(shade * (FX6_CEIL(y) - y)));

            for (i = 0; i < FX6_TRUNC(FX6_CEIL(w));
                 ++i, dst_cpy += item_stride) {
                for (b = 0; b < itemsize; ++b) {
                    dst_cpy[b] = 0;
                }
                dst_cpy[byteoffset] = edge_shade;
            }
        }

        for (j = 0; j < FX6_TRUNC(FX6_FLOOR(h + y) - FX6_CEIL(y)); ++j) {
            dst_cpy = dst;

            for (i = 0; i < FX6_TRUNC(FX6_CEIL(w));
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
            edge_shade =
                FX6_TRUNC(FX6_ROUND(shade * (h + y - FX6_FLOOR(h + y))));
            for (i = 0; i < FX6_TRUNC(FX6_CEIL(w));
                 ++i, dst_cpy += item_stride) {
                for (b = 0; b < itemsize; ++b) {
                    dst_cpy[b] = 0;
                }
                dst_cpy[byteoffset] = edge_shade;
            }
        }
    }
}

#ifndef NDEBUG
#define POINTER_ASSERT_DECLARATIONS(s)                               \
    const unsigned char *PA_bstart = ((unsigned char *)(s)->buffer); \
    const unsigned char *PA_bend = (PA_bstart + (s)->height * (s)->pitch);
#define POINTER_ASSERT(p)                            \
    assert((const unsigned char *)(p) >= PA_bstart); \
    assert((const unsigned char *)(p) < PA_bend);
#else
#define POINTER_ASSERT_DECLARATIONS(s)
#define POINTER_ASSERT(p)
#endif

#define _CREATE_RGB_FILLER(_bpp, _getp, _setp, _blendp)                       \
    void __fill_glyph_RGB##_bpp(FT_Fixed x, FT_Fixed y, FT_Fixed w,           \
                                FT_Fixed h, FontSurface *surface,             \
                                const FontColor *color)                       \
    {                                                                         \
        FT_Fixed dh = 0;                                                      \
        int i;                                                                \
        unsigned char *dst;                                                   \
        FT_UInt32 bgR, bgG, bgB, bgA;                                         \
        FT_Byte edge_a;                                                       \
        POINTER_ASSERT_DECLARATIONS(surface)                                  \
                                                                              \
        /* Crop the rectangle to the top and left of the                      \
         * surface.                                                           \
         */                                                                   \
        x = MAX(0, x);                                                        \
        y = MAX(0, y);                                                        \
                                                                              \
        /* Crop the rectangle to the bottom and right of                      \
         * the surface.                                                       \
         */                                                                   \
        if (x + w > INT_TO_FX6(surface->width)) {                             \
            w = INT_TO_FX6(surface->width) - x;                               \
        }                                                                     \
        if (y + h > INT_TO_FX6(surface->height)) {                            \
            h = INT_TO_FX6(surface->height) - y;                              \
        }                                                                     \
                                                                              \
        /* Start at the first pixel of the first row.                         \
         */                                                                   \
        dst = ((FT_Byte *)surface->buffer + FX6_TRUNC(FX6_CEIL(x)) * _bpp +   \
               FX6_TRUNC(FX6_CEIL(y)) * surface->pitch);                      \
                                                                              \
        /* Take care of the top row of the rectangle if the                   \
         * rectangle starts within the pixels: y is not on                    \
         * a pixel boundary. A special case is when the                       \
         * bottom of the rectangle is also with the pixel                     \
         * row.                                                               \
         */                                                                   \
        dh = FX6_CEIL(y) - y;                                                 \
        if (dh > h) {                                                         \
            dh = h;                                                           \
        }                                                                     \
        h -= dh;                                                              \
        if (dh > 0) {                                                         \
            unsigned char *_dst = dst - surface->pitch;                       \
                                                                              \
            edge_a = FX6_TRUNC(FX6_ROUND(color->a * dh));                     \
                                                                              \
            for (i = 0; i < FX6_TRUNC(FX6_CEIL(w)); ++i, _dst += _bpp) {      \
                FT_UInt32 pixel = (FT_UInt32)_getp;                           \
                                                                              \
                POINTER_ASSERT(_dst)                                          \
                                                                              \
                if (_bpp == 1) {                                              \
                    GET_PALETTE_VALS(pixel, surface->format, bgR, bgG, bgB,   \
                                     bgA);                                    \
                }                                                             \
                else {                                                        \
                    GET_RGB_VALS(pixel, surface->format, bgR, bgG, bgB, bgA); \
                }                                                             \
                                                                              \
                ALPHA_BLEND(color->r, color->g, color->b, edge_a, bgR, bgG,   \
                            bgB, bgA);                                        \
                                                                              \
                _blendp;                                                      \
            }                                                                 \
                                                                              \
            y += dh;                                                          \
        }                                                                     \
                                                                              \
        /* Fill in all entirely covered rows. These are                       \
         * pixels which are entirely within the upper and                     \
         * lower edges of the rectangle.                                      \
         */                                                                   \
        dh = FX6_FLOOR(h);                                                    \
        h -= dh;                                                              \
        while (dh > 0) {                                                      \
            unsigned char *_dst = dst;                                        \
                                                                              \
            for (i = 0; i < FX6_TRUNC(FX6_CEIL(w)); ++i, _dst += _bpp) {      \
                FT_UInt32 pixel = (FT_UInt32)_getp;                           \
                                                                              \
                POINTER_ASSERT(_dst)                                          \
                                                                              \
                if (_bpp == 1) {                                              \
                    GET_PALETTE_VALS(pixel, surface->format, bgR, bgG, bgB,   \
                                     bgA);                                    \
                }                                                             \
                else {                                                        \
                    GET_RGB_VALS(pixel, surface->format, bgR, bgG, bgB, bgA); \
                }                                                             \
                                                                              \
                ALPHA_BLEND(color->r, color->g, color->b, color->a, bgR, bgG, \
                            bgB, bgA);                                        \
                                                                              \
                _blendp;                                                      \
            }                                                                 \
                                                                              \
            dst += surface->pitch;                                            \
            dh -= FX6_ONE;                                                    \
            y += FX6_ONE;                                                     \
        }                                                                     \
                                                                              \
        /* Fill in the bottom row of pixels if these pixels                   \
         * are only partially covered: the rectangle bottom                   \
         * is not on a pixel boundary. Otherwise, done.                       \
         */                                                                   \
        if (h > 0) {                                                          \
            unsigned char *_dst = dst;                                        \
            edge_a = FX6_TRUNC(FX6_ROUND(color->a * h));                      \
                                                                              \
            for (i = 0; i < FX6_TRUNC(FX6_CEIL(w)); ++i, _dst += _bpp) {      \
                FT_UInt32 pixel = (FT_UInt32)_getp;                           \
                                                                              \
                POINTER_ASSERT(_dst)                                          \
                                                                              \
                if (_bpp == 1) {                                              \
                    GET_PALETTE_VALS(pixel, surface->format, bgR, bgG, bgB,   \
                                     bgA);                                    \
                }                                                             \
                else {                                                        \
                    GET_RGB_VALS(pixel, surface->format, bgR, bgG, bgB, bgA); \
                }                                                             \
                                                                              \
                ALPHA_BLEND(color->r, color->g, color->b, edge_a, bgR, bgG,   \
                            bgB, bgA);                                        \
                                                                              \
                _blendp;                                                      \
            }                                                                 \
        }                                                                     \
    }

#define __MONO_RENDER_INNER_LOOP(_bpp, _code)                  \
    for (j = ry; j < max_y; ++j) {                             \
        const unsigned char *_src = src;                       \
        unsigned char *_dst = dst;                             \
        FT_UInt32 val = (FT_UInt32)(*_src++ | 0x100) << shift; \
                                                               \
        for (i = rx; i < max_x; ++i, _dst += _bpp) {           \
            if (val & 0x10000) {                               \
                val = (FT_UInt32)(*_src++ | 0x100);            \
            }                                                  \
            if (val & 0x80) {                                  \
                _code;                                         \
            }                                                  \
            val <<= 1;                                         \
        }                                                      \
                                                               \
        src += bitmap->pitch;                                  \
        dst += surface->pitch;                                 \
    }

#define _CREATE_MONO_RENDER(_bpp, _getp, _setp, _blendp)                      \
    void __render_glyph_MONO##_bpp(int x, int y, FontSurface *surface,        \
                                   const FT_Bitmap *bitmap,                   \
                                   const FontColor *color)                    \
    {                                                                         \
        const int off_x = (x < 0) ? -x : 0;                                   \
        const int off_y = (y < 0) ? -y : 0;                                   \
                                                                              \
        const int max_x = MIN(x + (int)bitmap->width, (int)surface->width);   \
        const int max_y = MIN(y + (int)bitmap->rows, (int)surface->height);   \
                                                                              \
        const int rx = MAX(0, x);                                             \
        const int ry = MAX(0, y);                                             \
                                                                              \
        int i, j, shift;                                                      \
        const unsigned char *src;                                             \
        unsigned char *dst;                                                   \
        FT_UInt32 full_color;                                                 \
        FT_UInt32 bgR, bgG, bgB, bgA;                                         \
                                                                              \
        src = bitmap->buffer + (off_y * bitmap->pitch) + (off_x >> 3);        \
        dst = (unsigned char *)surface->buffer + (rx * _bpp) +                \
              (ry * surface->pitch);                                          \
                                                                              \
        full_color = SDL_MapRGBA(surface->format, (FT_Byte)color->r,          \
                                 (FT_Byte)color->g, (FT_Byte)color->b, 255);  \
                                                                              \
        shift = off_x & 7;                                                    \
                                                                              \
        if (color->a == 0xFF) {                                               \
            __MONO_RENDER_INNER_LOOP(_bpp, { _setp; });                       \
        }                                                                     \
        else if (color->a > 0) {                                              \
            __MONO_RENDER_INNER_LOOP(_bpp, {                                  \
                FT_UInt32 pixel = (FT_UInt32)_getp;                           \
                                                                              \
                if (_bpp == 1) {                                              \
                    GET_PALETTE_VALS(pixel, surface->format, bgR, bgG, bgB,   \
                                     bgA);                                    \
                }                                                             \
                else {                                                        \
                    GET_RGB_VALS(pixel, surface->format, bgR, bgG, bgB, bgA); \
                }                                                             \
                                                                              \
                ALPHA_BLEND(color->r, color->g, color->b, color->a, bgR, bgG, \
                            bgB, bgA);                                        \
                                                                              \
                _blendp;                                                      \
            });                                                               \
        }                                                                     \
    }

#define _CREATE_RGB_RENDER(_bpp, _getp, _setp, _blendp)                     \
    void __render_glyph_RGB##_bpp(int x, int y, FontSurface *surface,       \
                                  const FT_Bitmap *bitmap,                  \
                                  const FontColor *color)                   \
    {                                                                       \
        const int off_x = (x < 0) ? -x : 0;                                 \
        const int off_y = (y < 0) ? -y : 0;                                 \
                                                                            \
        const int max_x = MIN(x + (int)bitmap->width, (int)surface->width); \
        const int max_y = MIN(y + (int)bitmap->rows, (int)surface->height); \
                                                                            \
        const int rx = MAX(0, x);                                           \
        const int ry = MAX(0, y);                                           \
                                                                            \
        FT_Byte *dst = ((FT_Byte *)surface->buffer) + (rx * _bpp) +         \
                       (ry * surface->pitch);                               \
        FT_Byte *_dst;                                                      \
                                                                            \
        const FT_Byte *src =                                                \
            bitmap->buffer + off_x + (off_y * bitmap->pitch);               \
        const FT_Byte *_src;                                                \
                                                                            \
        _DECLARE_full_color##_bpp(                                          \
            surface,                                                        \
            color) /*                                                       \
                   const FT_UInt32 full_color =                             \
                       SDL_MapRGBA(surface->format, (FT_Byte)color->r,      \
                               (FT_Byte)color->g, (FT_Byte)color->b, 255);  \
                   */                                                       \
                                                                            \
            FT_UInt32 bgR,                                                  \
            bgG, bgB, bgA;                                                  \
        int j, i;                                                           \
                                                                            \
        for (j = ry; j < max_y; ++j) {                                      \
            _src = src;                                                     \
            _dst = dst;                                                     \
                                                                            \
            for (i = rx; i < max_x; ++i, _dst += _bpp) {                    \
                FT_UInt32 alpha = (*_src++);                                \
                alpha = (alpha * color->a) / 255;                           \
                                                                            \
                if (alpha == 0xFF) {                                        \
                    _setp;                                                  \
                }                                                           \
                else if (alpha > 0) {                                       \
                    FT_UInt32 pixel = (FT_UInt32)_getp;                     \
                                                                            \
                    if (_bpp == 1) {                                        \
                        GET_PALETTE_VALS(pixel, surface->format, bgR, bgG,  \
                                         bgB, bgA);                         \
                    }                                                       \
                    else {                                                  \
                        GET_RGB_VALS(pixel, surface->format, bgR, bgG, bgB, \
                                     bgA);                                  \
                    }                                                       \
                                                                            \
                    ALPHA_BLEND(color->r, color->g, color->b, alpha, bgR,   \
                                bgG, bgB, bgA);                             \
                                                                            \
                    _blendp;                                                \
                }                                                           \
            }                                                               \
                                                                            \
            dst += surface->pitch;                                          \
            src += bitmap->pitch;                                           \
        }                                                                   \
    }

/* These macros removes a gcc unused variable warning for __render_glyph_RGB3
 */
#define _DECLARE_full_color(s, c)             \
    const FT_UInt32 full_color = SDL_MapRGBA( \
        (s)->format, (FT_Byte)(c)->r, (FT_Byte)(c)->g, (FT_Byte)(c)->b, 255);
#define _DECLARE_full_color1(s, c) _DECLARE_full_color(s, c)
#define _DECLARE_full_color2(s, c) _DECLARE_full_color(s, c)
#define _DECLARE_full_color3(s, c)
#define _DECLARE_full_color4(s, c) _DECLARE_full_color(s, c)

#define _SET_PIXEL_24 \
    SET_PIXEL24_RGB(_dst, surface->format, color->r, color->g, color->b);

#define _BLEND_PIXEL_24 SET_PIXEL24_RGB(_dst, surface->format, bgR, bgG, bgB);

#define _SET_PIXEL(T) *(T *)_dst = (T)full_color;

#define _BLEND_PIXEL(T)                                                    \
    *((T *)_dst) =                                                         \
        (T)(((bgR >> surface->format->Rloss) << surface->format->Rshift) | \
            ((bgG >> surface->format->Gloss) << surface->format->Gshift) | \
            ((bgB >> surface->format->Bloss) << surface->format->Bshift) | \
            ((bgA >> surface->format->Aloss) << surface->format->Ashift &  \
             surface->format->Amask))

#define _BLEND_PIXEL_GENERIC(T)                                              \
    *(T *)_dst = (T)(SDL_MapRGB(surface->format, (FT_Byte)bgR, (FT_Byte)bgG, \
                                (FT_Byte)bgB))

#define _GET_PIXEL(T) (*((T *)_dst))

_CREATE_RGB_RENDER(4, _GET_PIXEL(FT_UInt32), _SET_PIXEL(FT_UInt32),
                   _BLEND_PIXEL(FT_UInt32))
_CREATE_RGB_RENDER(3, GET_PIXEL24(_dst), _SET_PIXEL_24, _BLEND_PIXEL_24)
_CREATE_RGB_RENDER(2, _GET_PIXEL(FT_UInt16), _SET_PIXEL(FT_UInt16),
                   _BLEND_PIXEL(FT_UInt16))
_CREATE_RGB_RENDER(1, _GET_PIXEL(FT_Byte), _SET_PIXEL(FT_Byte),
                   _BLEND_PIXEL_GENERIC(FT_Byte))

_CREATE_MONO_RENDER(4, _GET_PIXEL(FT_UInt32), _SET_PIXEL(FT_UInt32),
                    _BLEND_PIXEL(FT_UInt32))
_CREATE_MONO_RENDER(3, GET_PIXEL24(_dst), _SET_PIXEL_24, _BLEND_PIXEL_24)
_CREATE_MONO_RENDER(2, _GET_PIXEL(FT_UInt16), _SET_PIXEL(FT_UInt16),
                    _BLEND_PIXEL(FT_UInt16))
_CREATE_MONO_RENDER(1, _GET_PIXEL(FT_Byte), _SET_PIXEL(FT_Byte),
                    _BLEND_PIXEL_GENERIC(FT_Byte))

_CREATE_RGB_FILLER(4, _GET_PIXEL(FT_UInt32), _SET_PIXEL(FT_UInt32),
                   _BLEND_PIXEL(FT_UInt32))
_CREATE_RGB_FILLER(3, GET_PIXEL24(_dst), _SET_PIXEL_24, _BLEND_PIXEL_24)
_CREATE_RGB_FILLER(2, _GET_PIXEL(FT_UInt16), _SET_PIXEL(FT_UInt16),
                   _BLEND_PIXEL(FT_UInt16))
_CREATE_RGB_FILLER(1, _GET_PIXEL(FT_Byte), _SET_PIXEL(FT_Byte),
                   _BLEND_PIXEL_GENERIC(FT_Byte))
