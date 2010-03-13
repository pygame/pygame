/*
  pygame - Python Game Library
  Copyright (C) 2007-2010 Marcus von Appen

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

#include "surface.h"
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef HAVE_OPENMP
#define CREATE_FILLER(_name,_fillop)                                    \
    static int surface_fill_##_name(SDL_Surface *surface,               \
        SDL_Rect *rect, Uint32 color)                                   \
    {                                                                   \
        Uint8 *pixels, *ppx;                                            \
        int width = rect->w;                                            \
        int height = rect->h;                                           \
        int skip;                                                       \
        int bpp;                                                        \
        int n;                                                          \
        SDL_PixelFormat *fmt;                                           \
        Uint8 sR, sG, sB, sA, cR, cG, cB, cA;                           \
        Uint32 pixel;                                                   \
        Uint32 tmp;                                                     \
        int result = -1;                                                \
                                                                        \
        bpp = surface->format->BytesPerPixel;                           \
        fmt = surface->format;                                          \
        pixels = (Uint8 *) surface->pixels + surface->offset +          \
            (Uint16) rect->y * surface->pitch + (Uint16) rect->x * bpp; \
        skip = surface->pitch - width * bpp;                            \
                                                                        \
        switch (bpp)                                                    \
        {                                                               \
        case 1:                                                         \
        {                                                               \
            int x, y;                                                   \
            SDL_GetRGBA (color, fmt, &cR, &cG, &cB, &cA);               \
            _Pragma("omp parallel")                                     \
            {                                                           \
                _Pragma("omp for private(ppx,x,sR,sG,sB,sA,tmp)")       \
                for (y = 0; y < height; y++)                            \
                {                                                       \
                    for (x = 0; x < width; x++)                         \
                    {                                                   \
                        ppx = pixels + y * surface->pitch + x * bpp;    \
                        GET_PALETTE_VALS(ppx,fmt,sR,sG,sB,sA);          \
                        _fillop;                                        \
                        *ppx = SDL_MapRGBA(fmt,sR,sG,sB,sA);            \
                    }                                                   \
                }                                                       \
            }                                                           \
            result = 0;                                                 \
            break;                                                      \
        }                                                               \
        default:                                                        \
        {                                                               \
            int x, y;                                                   \
            GET_RGB_VALS (color, fmt, cR, cG, cB, cA);                  \
            _Pragma("omp parallel")                                     \
            {                                                           \
                _Pragma("omp for private(ppx,pixel,x,sR,sG,sB,sA,tmp)") \
                for (y = 0; y < height; y++)                            \
                {                                                       \
                    for (x = 0; x < width; x++)                         \
                    {                                                   \
                        ppx = pixels + y * surface->pitch + x * bpp;    \
                        GET_PIXEL(pixel,bpp,ppx);                       \
                        GET_RGB_VALS(pixel,fmt,sR,sG,sB,sA);            \
                        _fillop;                                        \
                        CREATE_PIXEL(ppx,sR,sG,sB,sA,bpp,fmt);          \
                    }                                                   \
                }                                                       \
            }                                                           \
            result = 0;                                                 \
            break;                                                      \
        }                                                               \
        }                                                               \
        return result;                                                  \
    }
#else /* HAVE_OPENMP */
#define CREATE_FILLER(_name,_fillop)                                    \
    static int surface_fill_##_name(SDL_Surface *surface,               \
        SDL_Rect *rect, Uint32 color)                                   \
    {                                                                   \
        Uint8 *pixels;                                                  \
        int width = rect->w;                                            \
        int height = rect->h;                                           \
        int skip;                                                       \
        int bpp;                                                        \
        int n;                                                          \
        SDL_PixelFormat *fmt;                                           \
        Uint8 sR, sG, sB, sA, cR, cG, cB, cA;                           \
        Uint32 pixel;                                                   \
        Uint32 tmp;                                                     \
        int result = -1;                                                \
                                                                        \
        bpp = surface->format->BytesPerPixel;                           \
        fmt = surface->format;                                          \
        pixels = (Uint8 *) surface->pixels + surface->offset +          \
            (Uint16) rect->y * surface->pitch + (Uint16) rect->x * bpp; \
        skip = surface->pitch - width * bpp;                            \
                                                                        \
        switch (bpp)                                                    \
        {                                                               \
        case 1:                                                         \
        {                                                               \
            SDL_GetRGBA (color, fmt, &cR, &cG, &cB, &cA);               \
            while (height--)                                            \
            {                                                           \
                LOOP_UNROLLED4(                                         \
                {                                                       \
                    GET_PALETTE_VALS (pixels, fmt, sR, sG, sB, sA);     \
                    _fillop;                                            \
                    *pixels = SDL_MapRGBA (fmt, sR, sG, sB, sA);        \
                    pixels += bpp;                                      \
                }, n, width);                                           \
                pixels += skip;                                         \
            }                                                           \
            result = 0;                                                 \
            break;                                                      \
        }                                                               \
        default:                                                        \
        {                                                               \
            GET_RGB_VALS (color, fmt, cR, cG, cB, cA);                  \
            while (height--)                                            \
            {                                                           \
                LOOP_UNROLLED4(                                         \
                {                                                       \
                    GET_PIXEL (pixel, bpp, pixels);                     \
                    GET_RGB_VALS (pixel, fmt, sR, sG, sB, sA);          \
                    _fillop;                                            \
                    CREATE_PIXEL(pixels, sR, sG, sB, sA, bpp, fmt);     \
                    pixels += bpp;                                      \
                }, n, width);                                           \
                pixels += skip;                                         \
            }                                                           \
            result = 0;                                                 \
            break;                                                      \
        }                                                               \
        }                                                               \
        return result;                                                  \
    }
#endif /* HAVE_OPENMP */

CREATE_FILLER(blend_rgba_add, D_BLEND_RGBA_ADD(tmp,cR,cG,cB,cA,sR,sG,sB,sA))
CREATE_FILLER(blend_rgba_sub, D_BLEND_RGBA_SUB(tmp,cR,cG,cB,cA,sR,sG,sB,sA))
CREATE_FILLER(blend_rgba_mult, D_BLEND_RGBA_MULT(cR,cG,cB,cA,sR,sG,sB,sA))
CREATE_FILLER(blend_rgba_min, D_BLEND_RGBA_MIN(cR,cG,cB,cA,sR,sG,sB,sA))
CREATE_FILLER(blend_rgba_max, D_BLEND_RGBA_MAX(cR,cG,cB,cA,sR,sG,sB,sA))
CREATE_FILLER(blend_rgba_and, D_BLEND_RGBA_AND(cR,cG,cB,cA,sR,sG,sB,sA))
CREATE_FILLER(blend_rgba_or, D_BLEND_RGBA_OR(cR,cG,cB,cA,sR,sG,sB,sA))
CREATE_FILLER(blend_rgba_xor, D_BLEND_RGBA_XOR(cR,cG,cB,cA,sR,sG,sB,sA))
CREATE_FILLER(blend_rgba_diff, D_BLEND_RGBA_DIFF(cR,cG,cB,cA,sR,sG,sB,sA))
CREATE_FILLER(blend_rgba_screen, D_BLEND_RGBA_SCREEN(cR,cG,cB,cA,sR,sG,sB,sA))
CREATE_FILLER(blend_rgba_avg, D_BLEND_RGBA_AVG(cR,cG,cB,cA,sR,sG,sB,sA))
CREATE_FILLER(blend_rgb_add, D_BLEND_RGB_ADD(tmp,cR,cG,cB,sR,sG,sB))
CREATE_FILLER(blend_rgb_sub, D_BLEND_RGB_SUB(tmp,cR,cG,cB,sR,sG,sB))
CREATE_FILLER(blend_rgb_mult, D_BLEND_RGB_MULT(cR,cG,cB,sR,sG,sB))
CREATE_FILLER(blend_rgb_min, D_BLEND_RGB_MIN(cR,cG,cB,sR,sG,sB))
CREATE_FILLER(blend_rgb_max, D_BLEND_RGB_MAX(cR,cG,cB,sR,sG,sB))
CREATE_FILLER(blend_rgb_and, D_BLEND_RGB_AND(cR,cG,cB,sR,sG,sB))
CREATE_FILLER(blend_rgb_or, D_BLEND_RGB_OR(cR,cG,cB,sR,sG,sB))
CREATE_FILLER(blend_rgb_xor, D_BLEND_RGB_XOR(cR,cG,cB,sR,sG,sB))
CREATE_FILLER(blend_rgb_avg, D_BLEND_RGB_AVG(cR,cG,cB,sR,sG,sB))
CREATE_FILLER(blend_rgb_diff, D_BLEND_RGB_DIFF(cR,cG,cB,sR,sG,sB))
CREATE_FILLER(blend_rgb_screen, D_BLEND_RGB_SCREEN(cR,cG,cB,sR,sG,sB))

int
pyg_sdlsurface_fill_blend (SDL_Surface *surface, SDL_Rect *rect, Uint32 color,
    int blendargs)
{
    int result = -1;
    int locked = 0;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (!rect)
    {
        SDL_SetError ("rect argument NULL");
        return 0;
    }

    /* Lock the surface, if needed */
    if (SDL_MUSTLOCK (surface))
    {
        if (SDL_LockSurface (surface) < 0)
            return -1;
        locked = 1;
    }

    switch (blendargs)
    {
    case BLEND_RGB_ADD:
    {
        result = surface_fill_blend_rgb_add (surface, rect, color);
        break;
    }
    case BLEND_RGB_SUB:
    {
        result = surface_fill_blend_rgb_sub (surface, rect, color);
        break;
    }
    case BLEND_RGB_MULT:
    {
        result = surface_fill_blend_rgb_mult (surface, rect, color);
        break;
    }
    case BLEND_RGB_MIN:
    {
        result = surface_fill_blend_rgb_min (surface, rect, color);
        break;
    }
    case BLEND_RGB_MAX:
    {
        result = surface_fill_blend_rgb_max (surface, rect, color);
        break;
    }
    case BLEND_RGB_AND:
    {
        result = surface_fill_blend_rgb_and (surface, rect, color);
        break;
    }
    case BLEND_RGB_OR:
    {
        result = surface_fill_blend_rgb_or (surface, rect, color);
        break;
    }
    case BLEND_RGB_XOR:
    {
        result = surface_fill_blend_rgb_xor (surface, rect, color);
        break;
    }
    case BLEND_RGB_DIFF:
    {
        result = surface_fill_blend_rgb_diff (surface, rect, color);
        break;
    }
    case BLEND_RGB_SCREEN:
    {
        result = surface_fill_blend_rgb_screen (surface, rect, color);
        break;
    }
    case BLEND_RGB_AVG:
    {
        result = surface_fill_blend_rgb_avg (surface, rect, color);
        break;
    }

    case BLEND_RGBA_ADD:
    {
        result = surface_fill_blend_rgba_add (surface, rect, color);
        break;
    }
    case BLEND_RGBA_SUB:
    {
        result = surface_fill_blend_rgba_sub (surface, rect, color);
        break;
    }
    case BLEND_RGBA_MULT:
    {
        result = surface_fill_blend_rgba_mult (surface, rect, color);
        break;
    }
    case BLEND_RGBA_MIN:
    {
        result = surface_fill_blend_rgba_min (surface, rect, color);
        break;
    }
    case BLEND_RGBA_MAX:
    {
        result = surface_fill_blend_rgba_max (surface, rect, color);
        break;
    }
        case BLEND_RGBA_AND:
    {
        result = surface_fill_blend_rgba_and (surface, rect, color);
        break;
    }
    case BLEND_RGBA_OR:
    {
        result = surface_fill_blend_rgba_or (surface, rect, color);
        break;
    }
    case BLEND_RGBA_XOR:
    {
        result = surface_fill_blend_rgba_xor (surface, rect, color);
        break;
    }
    case BLEND_RGBA_DIFF:
    {
        result = surface_fill_blend_rgba_diff (surface, rect, color);
        break;
    }
    case BLEND_RGBA_SCREEN:
    {
        result = surface_fill_blend_rgba_screen (surface, rect, color);
        break;
    }
    case BLEND_RGBA_AVG:
    {
        result = surface_fill_blend_rgba_avg (surface, rect, color);
        break;
    }
    
    default:
    {
        result = -1;
        SDL_SetError ("invalid blend argument");
        break;
    }
    }

    if (locked)
        SDL_UnlockSurface (surface);

    return result;
}
