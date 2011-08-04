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

typedef FT_Byte _T1;
typedef FT_UInt16 _T2;
typedef FT_UInt32 _T3;
typedef FT_UInt32 _T4;

void __render_glyph_GRAY1(int x, int y, FaceSurface *surface,
                          FT_Bitmap *bitmap, FaceColor *color)
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

void __render_glyph_MONO_as_GRAY1(int x, int y, FaceSurface *surface,
                                  FT_Bitmap *bitmap, FaceColor *color)
{
    const int off_x = (x < 0) ? -x : 0;
    const int off_y = (y < 0) ? -y : 0;

    const int max_x = MIN(x + bitmap->width, surface->width);
    const int max_y = MIN(y + bitmap->rows, surface->height);

    const int rx = MAX(0, x);
    const int ry = MAX(0, y);

    int             i, j, shift;
    unsigned char*  src;
    unsigned char*  dst;
    unsigned char*  src_cpy;
    unsigned char*  dst_cpy;
    FT_UInt32       val;
    FT_Byte shade = color->a;

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

void __render_glyph_GRAY_as_MONO1(int x, int y, FaceSurface *surface,
                                  FT_Bitmap *bitmap, FaceColor *color)
{
    FT_Byte *dst = ((FT_Byte *)surface->buffer) + x + (y * surface->pitch);
    FT_Byte *dst_cpy;
    FT_Byte shade = color->a;

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

void __fill_glyph_GRAY1(int x, int y, int w, int h, FaceSurface *surface,
                        FaceColor *color)
{
    int i, j;
    FT_Byte *dst;
    FT_Byte *dst_cpy;
    FT_Byte shade = color->a;

    x = MAX(0, x);
    y = MAX(0, y);

    if (x + w > surface->width) {
        w = surface->width - x;
    }
    if (y + h > surface->height) {
        h = surface->height - y;
    }

    dst = (FT_Byte *)surface->buffer + x + (y * surface->pitch);

    for (j = 0; j < h; ++j) {
        dst_cpy = dst;

        for (i = 0; i < w; ++i, ++dst_cpy) {
            *dst_cpy = shade;
        }

        dst += surface->pitch;
    }
}

#ifdef HAVE_PYGAME_SDL_VIDEO

#define _CREATE_RGB_FILLER(_bpp)                            \
    void __fill_glyph_RGB##_bpp(int x, int y, int w, int h, \
                                FaceSurface *surface,       \
                                FaceColor *color)           \
    {                                                       \
        SDL_PixelFormat *format = surface->format;          \
        FT_UInt32 bgR, bgG, bgB, bgA;                       \
        int i, j;                                           \
        unsigned char *dst;                                 \
                                                            \
        if (color->a == 0) {                                \
            /* Nothing to do. */                            \
            return;                                         \
        }                                                   \
        x = MAX(0, x);                                      \
        y = MAX(0, y);                                      \
                                                            \
        if (x + w > surface->width) {                       \
            w = surface->width - x;                         \
        }                                                   \
        if (y + h > surface->height) {                      \
            h = surface->height - y;                        \
        }                                                   \
                                                            \
        dst = ((unsigned char *)surface->buffer +           \
               (x * _bpp) +                                 \
               (y * surface->pitch));                       \
                                                            \
        for (j = 0; j < h; ++j) {                           \
            unsigned char *_dst = dst;                      \
                                                            \
            for (i = 0; i < w; ++i, _dst += _bpp) {         \
                FT_UInt32 pixel =                           \
                    (FT_UInt32)_GET_PIXEL(_bpp, _dst);      \
                                                            \
                _UNMAP_PIXEL(_bpp, bgR, bgG, bgB, bgA,      \
                             format, pixel);                \
                                                            \
                ALPHA_BLEND(color->r, color->g,             \
                            color->b, color->a,             \
                            bgR, bgG, bgB, bgA);            \
                                                            \
                _MAP_PIXEL(_bpp, pixel, format,             \
                           bgR, bgG, bgB, bgA);             \
                _SET_PIXEL(_bpp, _dst, pixel);              \
            }                                               \
                                                            \
            dst += surface->pitch;                          \
        }                                                   \
    }

#define __MONO_RENDER_INNER_LOOP(_bpp, _code)               \
    for (j = ry; j < max_y; ++j)                            \
    {                                                       \
        unsigned char* _src = src;                          \
        unsigned char* _dst = dst;                          \
        FT_UInt32 val =                                     \
            (FT_UInt32)(*_src++ | 0x100) << shift;          \
                                                            \
        for (i = rx; i < max_x; ++i, _dst += _bpp) {        \
            if (val & 0x10000) {                            \
                val = (FT_UInt32)(*_src++ | 0x100);         \
            }                                               \
            _code(_bpp, val & 0x80);                        \
            val <<= 1;                                      \
        }                                                   \
                                                            \
        src += bitmap->pitch;                               \
        dst += surface->pitch;                              \
    }                                                       \

#define __MONO_RENDER_PIXEL_OPAQUE(_bpp, is_foreground)     \
    if (is_foreground) {                                    \
        _SET_PIXEL(_bpp, _dst, full_color);                 \
    }

#define __MONO_RENDER_PIXEL_ALPHA(_bpp, is_foreground)        \
    if (is_foreground) {                                      \
        FT_UInt32 pixel =                                     \
            (FT_UInt32)_GET_PIXEL(_bpp, _dst);                \
                                                              \
        _UNMAP_PIXEL(_bpp, bgR, bgG, bgB, bgA,                \
                     format, pixel);                          \
        ALPHA_BLEND(fgR_full, fgG_full,                       \
                    fgB_full, fgA_full,                       \
                    bgR, bgG, bgB, bgA);                      \
        _MAP_PIXEL(_bpp, pixel, format, bgR, bgG, bgB, bgA);  \
        _SET_PIXEL(_bpp, _dst, pixel);                        \
    }

#define _CREATE_MONO_RENDER(_bpp)                           \
    void __render_glyph_MONO##_bpp(int x, int y,            \
                                   FaceSurface *surface,    \
                                   FT_Bitmap *bitmap,       \
                                   FaceColor *color)        \
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
        SDL_PixelFormat *format = surface->format;                      \
        int i, j, shift;                                                \
        unsigned char *src;                                             \
        unsigned char *dst;                                             \
        FT_UInt32 fgR_full = color->r;                                  \
        FT_UInt32 fgG_full = color->g;                                  \
        FT_UInt32 fgB_full = color->b;                                  \
        FT_UInt32 fgA_full = color->a;                                  \
        _T(_bpp) full_color;                                            \
        FT_UInt32 bgR, bgG, bgB, bgA;                                   \
                                                                        \
        src  = bitmap->buffer + (off_y * bitmap->pitch) + (off_x >> 3); \
        dst = (unsigned char *)surface->buffer + (rx * _bpp) +          \
                    (ry * surface->pitch);                              \
                                                                        \
        _MAP_PIXEL(_bpp, full_color, format,                            \
                   fgR_full, fgG_full, fgB_full, fgA_full);             \
                                                                        \
        shift = off_x & 7;                                              \
                                                                        \
        if (color->a == SDL_ALPHA_OPAQUE) {                             \
            __MONO_RENDER_INNER_LOOP(_bpp, __MONO_RENDER_PIXEL_OPAQUE); \
        }                                                               \
        else if (color->a > SDL_ALPHA_TRANSPARENT) {                    \
            __MONO_RENDER_INNER_LOOP(_bpp, __MONO_RENDER_PIXEL_ALPHA);  \
        }                                                               \
    }

#define _CREATE_RGB_RENDER(_bpp)                                        \
    void __render_glyph_RGB##_bpp(int x, int y, FaceSurface *surface,   \
                                  FT_Bitmap *bitmap, FaceColor *color)  \
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
        SDL_PixelFormat *format = surface->format;                      \
        FT_Byte *dst = ((FT_Byte*)surface->buffer) + (rx * _bpp) +      \
                        (ry * surface->pitch);                          \
        FT_Byte *_dst;                                                  \
                                                                        \
        const FT_Byte *src = bitmap->buffer + off_x +                   \
                                (off_y * bitmap->pitch);                \
        const FT_Byte *_src;                                            \
                                                                        \
        FT_UInt32 bgR, bgG, bgB, bgA;                                   \
        FT_UInt32 fgR_full = color->r;                                  \
        FT_UInt32 fgG_full = color->g;                                  \
        FT_UInt32 fgB_full = color->b;                                  \
        FT_UInt32 fgA_full = color->a;                                  \
        _T(_bpp) full_color;                                            \
        int j, i;                                                       \
                                                                        \
        _MAP_PIXEL(_bpp, full_color, format,                            \
                   fgR_full, fgG_full, fgB_full, fgA_full);             \
                                                                        \
        for (j = ry; j < max_y; ++j) {                                  \
            _src = src;                                                 \
            _dst = dst;                                                 \
                                                                        \
            for (i = rx; i < max_x; ++i, _dst += _bpp) {                \
                FT_UInt32 alpha = (*_src++) * fgA_full / 255;           \
                                                                        \
                if (alpha == SDL_ALPHA_OPAQUE) {                        \
                    _SET_PIXEL(_bpp, _dst, full_color);                 \
                }                                                       \
                else if (alpha != SDL_ALPHA_TRANSPARENT) {              \
                    FT_UInt32 pixel =                                   \
                        (FT_UInt32)_GET_PIXEL(_bpp, _dst);              \
                                                                        \
                    _UNMAP_PIXEL(_bpp, bgR, bgG, bgB, bgA,              \
                                 format, pixel);                        \
                                                                        \
                    if (bgA == 0) {                                     \
                        _SET_PIXEL(_bpp, _dst, full_color);             \
                    }                                                   \
                    else {                                              \
                        ALPHA_BLEND(fgR_full, fgG_full, fgB_full,       \
                                    alpha,                              \
                                    bgR, bgG, bgB, bgA);                \
                        _MAP_PIXEL(_bpp, pixel, format,                 \
                                   bgR, bgG, bgB, bgA);                 \
                        _SET_PIXEL(_bpp, _dst, pixel);                  \
                    }                                                   \
                }                                                       \
            }                                                           \
                                                                        \
            dst += surface->pitch;                                      \
            src += bitmap->pitch;                                       \
        }                                                               \
    }

#define _T(_bpp) _T##_bpp

#define _GET_PIXEL(_bpp, _sp) _GET_PIXEL##_bpp(_sp)
#define _GET_PIXELT(_T, _sp) (*(_T *)(_sp))
#define _GET_PIXEL1(_sp) _GET_PIXELT(_T1, _sp)
#define _GET_PIXEL2(_sp) _GET_PIXELT(_T2, _sp)
#define _GET_PIXEL3(_sp) GET_PIXEL24((FT_Byte *)(_sp))
#define _GET_PIXEL4(_sp) _GET_PIXELT(_T4, _sp)

#define _SET_PIXEL(_bpp, _bufp, _s) _SET_PIXEL##_bpp(_bufp, _s)
#define _SET_PIXELT(_T, _bufp, _s) (*(_T *)(_bufp)) = (_T)(_s)
#define _SET_PIXEL1(_bufp, _s) _SET_PIXELT(_T1, _bufp, _s)
#define _SET_PIXEL2(_bufp, _s) _SET_PIXELT(_T2, _bufp, _s)
#define _SET_PIXEL3(_bufp, _s) SET_PIXEL24((FT_Byte *)(_bufp), _s)
#define _SET_PIXEL4(_bufp, _s) _SET_PIXELT(_T4, _bufp, _s)

#define _UNMAP_PIXEL(_bpp, _r, _g, _b, _a, _fmtp, _pix) \
    _UNMAP_PIXEL##_bpp(_r, _g, _b, _a, _fmtp, _pix)
#define _UNMAP_PIXEL1(_r, _g, _b, _a, _fmtp, _i)        \
    UNMAP_PALETTE_INDEX(_i, _fmtp, _r, _g, _b, _a)
#define _UNMAP_PIXEL2(_r, _g, _b, _a, _fmtp, _pix)      \
    UNMAP_RGB_VALUE(_pix, _fmtp, _r, _g, _b, _a)
#define _UNMAP_PIXEL3(_r, _g, _b, _a, _fmtp, _pix)      \
    UNMAP_RGB_VALUE(_pix, _fmtp, _r, _g, _b, _a)
#define _UNMAP_PIXEL4(_r, _g, _b, _a, _fmtp, _pix)      \
    UNMAP_RGB_VALUE(_pix, _fmtp, _r, _g, _b, _a)

#define _MAP_PIXEL(_bpp, _pix, _fmtp, _r, _g, _b, _a)  \
    _MAP_PIXEL##_bpp(_pix, _fmtp, _r, _g, _b, _a)
#define _MAP_PIXELT(_T, _pix, _fmtp, _r, _g, _b, _a)    \
    do {                                                \
        _pix = (_T)(                                    \
            (((_r) >> _fmtp->Rloss) << _fmtp->Rshift) | \
            (((_g) >> _fmtp->Gloss) << _fmtp->Gshift) | \
            (((_b) >> _fmtp->Bloss) << _fmtp->Bshift) | \
            (((_a) >> _fmtp->Aloss) << _fmtp->Ashift  & \
             surface->format->Amask));                  \
    }                                                   \
    while (0)
#define _MAP_PIXEL_GENERIC(_T, _pix, _fmtp, _r, _g, _b, _a)             \
    do {                                                                \
        _pix = (_T)(SDL_MapRGB((_fmtp), (FT_Byte)(_r),                  \
                               (FT_Byte)(_g), (FT_Byte)(_b)));          \
    }                                                                   \
    while (0)
#define _MAP_PIXEL1(_pix, _fmtp, _r, _g, _b, _a)                \
    _MAP_PIXEL_GENERIC(_T1, _pix, _fmtp, _r, _g, _b, _a)
#define _MAP_PIXEL2(_pix, _fmtp, _r, _g, _b, _a)        \
    _MAP_PIXELT(_T2, _pix, _fmtp, _r, _g, _b, _a)
#define _MAP_PIXEL3(_pix, _fmtp, _r, _g, _b, _a)                \
    MAP_PIXEL24((FT_Byte *)(&(_pix)), _fmtp, _r, _g, _b)
#define _MAP_PIXEL4(_pix, _fmtp, _r, _g, _b, _a)        \
    _MAP_PIXELT(_T4, _pix, _fmtp, _r, _g, _b, _a)

_CREATE_RGB_RENDER(4)
_CREATE_RGB_RENDER(3)
_CREATE_RGB_RENDER(2)
_CREATE_RGB_RENDER(1)

_CREATE_MONO_RENDER(4)
_CREATE_MONO_RENDER(3)
_CREATE_MONO_RENDER(2)
_CREATE_MONO_RENDER(1)

_CREATE_RGB_FILLER(4)
_CREATE_RGB_FILLER(3)
_CREATE_RGB_FILLER(2)
_CREATE_RGB_FILLER(1)
#endif
