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

#define GET_RGB_VALS(pixel, fmt, r, g, b, a)                            \
    (r) = ((pixel) & (fmt)->Rmask) >> (fmt)->Rshift;                    \
    (r) = ((r) << (fmt)->Rloss) + ((r) >> (8 - ((fmt)->Rloss << 1)));   \
    (g) = ((pixel) & (fmt)->Gmask) >> (fmt)->Gshift;                    \
    (g) = ((g) << (fmt)->Gloss) + ((g) >> (8 - ((fmt)->Gloss << 1)));   \
    (b) = ((pixel) & (fmt)->Bmask) >> (fmt)->Bshift;                    \
    (b) = ((b) << (fmt)->Bloss) + ((b) >> (8 - ((fmt)->Bloss << 1)));   \
    if ((fmt)->Amask)                                                   \
    {                                                                   \
        (a) = ((pixel) & (fmt)->Amask) >> (fmt)->Ashift;                \
        (a) = ((a) << (fmt)->Aloss) + ((a) >> (8 - ((fmt)->Aloss << 1))); \
    }                                                                   \
    else                                                                \
        (a) = 255;

#define GET_PALETTE_VALS(pixel, fmt, sr, sg, sb, sa)                    \
    (sr) = (fmt)->palette->colors[*((Uint8 *) (pixel))].r;              \
    (sg) = (fmt)->palette->colors[*((Uint8 *) (pixel))].g;              \
    (sb) = (fmt)->palette->colors[*((Uint8 *) (pixel))].b;              \
    (sa) = 255;

#define GET_PIXEL_VALS(pixel, fmt, r, g, b, a)          \
    if ((fmt)->palette == NULL)                         \
    {                                                   \
        GET_RGB_VALS(pixel, fmt, r, g, b, a);           \
    }                                                   \
    else                                                \
    {                                                   \
        GET_PALETTE_VALS (pixel, fmt, r, g, b, a);      \
    }

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define GET_PIXEL24(b) ((b)[0] + ((b)[1] << 8) + ((b)[2] << 16))
#define SET_PIXEL24_RGB(buf,format,r,g,b)                               \
    *((buf) + ((format)->Rshift >> 3)) = (r);                           \
    *((buf) + ((format)->Gshift >> 3)) = (g);                           \
    *((buf) + ((format)->Bshift >> 3)) = (b);
#define SET_PIXEL24(buf,format,rgb)                                     \
    *((buf) + ((format)->Rshift >> 3)) = (rgb)[0];                      \
    *((buf) + ((format)->Gshift >> 3)) = (rgb)[1];                      \
    *((buf) + ((format)->Bshift >> 3)) = (rgb)[2];
#else
#define GET_PIXEL24(b) ((b)[2] + ((b)[1] << 8) + ((b)[0] << 16))
#define SET_PIXEL24_RGB(buf,format,r,g,b)                               \
    *((buf) + 2 - ((format)->Rshift >> 3)) = (r);                       \
    *((buf) + 2 - ((format)->Gshift >> 3)) = (g);                       \
    *((buf) + 2 - ((format)->Bshift >> 3)) = (b);
#define SET_PIXEL24(buf,format,rgb)                                     \
    *((buf) + 2 - ((format)->Rshift >> 3)) = (rgb)[0];                  \
    *((buf) + 2 - ((format)->Gshift >> 3)) = (rgb)[1];                  \
    *((buf) + 2 - ((format)->Bshift >> 3)) = (rgb)[2];
#endif

#endif
