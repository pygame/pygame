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
#ifndef _PYGAME_FREETYPE_PIXEL_H_
#define _PYGAME_FREETYPE_PIXEL_H_

#include "../surface.h"

#define UNMAP_RGB_VALUE(pixel, fmt, r, g, b, a)                         \
    (r) = ((pixel) & (fmt)->Rmask) >> (fmt)->Rshift;                    \
    (r) = ((r) << (fmt)->Rloss) + ((r) >> (8 - ((fmt)->Rloss << 1)));   \
    (g) = ((pixel) & (fmt)->Gmask) >> (fmt)->Gshift;                    \
    (g) = ((g) << (fmt)->Gloss) + ((g) >> (8 - ((fmt)->Gloss << 1)));   \
    (b) = ((pixel) & (fmt)->Bmask) >> (fmt)->Bshift;                    \
    (b) = ((b) << (fmt)->Bloss) + ((b) >> (8 - ((fmt)->Bloss << 1)));   \
    if ((fmt)->Amask) {                                                 \
        (a) = ((pixel) & (fmt)->Amask) >> (fmt)->Ashift;                \
        (a) = ((a) << (fmt)->Aloss) +                                   \
               ((a) >> (8 - ((fmt)->Aloss << 1)));                      \
    }                                                                   \
    else {                                                              \
        (a) = 255;                                                      \
    }

#define UNMAP_PALETTE_INDEX(pixel, fmt, sr, sg, sb, sa)                 \
    (sr) = (fmt)->palette->colors[(Uint8) (pixel)].r;                   \
    (sg) = (fmt)->palette->colors[(Uint8) (pixel)].g;                   \
    (sb) = (fmt)->palette->colors[(Uint8) (pixel)].b;                   \
    (sa) = 255;

#define GET_PIXEL_VALS(pixel, fmt, r, g, b, a)          \
    if (!(fmt)->palette) {                              \
        GET_RGB_VALS(pixel, fmt, r, g, b, a);           \
    }                                                   \
    else {                                              \
        GET_PALETTE_VALS (pixel, fmt, r, g, b, a);      \
    }

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define GET_PIXEL24(bufp) ((bufp)[0] + ((bufp)[1] << 8) + ((bufp)[2] << 16))
#define MAP_PIXEL24(bufp, format, r, g, b)                              \
    do {                                                                \
        (bufp)[(format)->Rshift >> 3] = (r);                            \
        (bufp)[(format)->Gshift >> 3] = (g);                            \
        (bufp)[(format)->Bshift >> 3] = (b);                            \
    }                                                                   \
    while (0)
#define SET_PIXEL24(bufp, pix)                  \
    do {                                        \
        (bufp)[0] = (FT_Byte)(pix);             \
        (bufp)[1] = (FT_Byte)(pix >> 8);        \
        (bufp)[2] = (FT_Byte)(pix >> 16);       \
    }                                           \
    while (0)
#else
#define GET_PIXEL24(bufp) ((bufp)[2] + ((bufp)[1] << 8) + ((bufp)[0] << 16))
#define MAP_PIXEL24(bufp, format, r, g, b)                              \
    do {                                                                \
        (bufp)[2 - (format)->Rshift >> 3] = (r);                        \
        (bufp)[2 - (format)->Gshift >> 3] = (g);                        \
        (bufp)[2 - (format)->Bshift >> 3] = (b);                        \
    }                                                                   \
    while (0)
#define SET_PIXEL24(bufp, pix)                  \
    do {                                        \
        (bufp)[2] = (FT_Byte)(pix);             \
        (bufp)[1] = (FT_Byte)(pix >> 8);        \
        (bufp)[0] = (FT_Byte)(pix >> 16);       \
    }                                           \
    while (0)
#endif

#endif
