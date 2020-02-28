/*
  pygame - Python Game Library
  Copyright (C) 2007 Marcus von Appen

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

#define NO_PYGAME_C_API
#include "_surface.h"

/*
 * Changes SDL_Rect to respect any clipping rect defined on the surface.
 * Necessary when modifying surface->pixels directly instead of through an
 * SDL interface.
 */
void
surface_respect_clip_rect(SDL_Surface *surface, SDL_Rect *rect)
{
    SDL_Rect tmp;
    SDL_Rect *A, *B;
    int x, y, w, h;

    A = rect;
    B = &tmp;
    SDL_GetClipRect(surface, B);

    /* Code here is nearly identical to rect_clip in rect.c */

    /* Left */
    if ((A->x >= B->x) && (A->x < (B->x + B->w)))
        x = A->x;
    else if ((B->x >= A->x) && (B->x < (A->x + A->w)))
        x = B->x;
    else
        return;

    /* Right */
    if (((A->x + A->w) > B->x) && ((A->x + A->w) <= (B->x + B->w)))
        w = (A->x + A->w) - x;
    else if (((B->x + B->w) > A->x) && ((B->x + B->w) <= (A->x + A->w)))
        w = (B->x + B->w) - x;
    else
        return;

    /* Top */
    if ((A->y >= B->y) && (A->y < (B->y + B->h)))
        y = A->y;
    else if ((B->y >= A->y) && (B->y < (A->y + A->h)))
        y = B->y;
    else
        return;

    /* Bottom */
    if (((A->y + A->h) > B->y) && ((A->y + A->h) <= (B->y + B->h)))
        h = (A->y + A->h) - y;
    else if (((B->y + B->h) > A->y) && ((B->y + B->h) <= (A->y + A->h)))
        h = (B->y + B->h) - y;
    else
        return;

    rect->x = x;
    rect->y = y;
    rect->w = w;
    rect->h = h;
}

static int
surface_fill_blend_add(SDL_Surface *surface, SDL_Rect *rect, Uint32 color)
{
    Uint8 *pixels;
    int width = rect->w;
    int height = rect->h;
    int skip;
    int bpp = surface->format->BytesPerPixel;
    int n;
    SDL_PixelFormat *fmt = surface->format;
    Uint8 sR, sG, sB, sA, cR, cG, cB, cA;
    Uint32 pixel;
    Uint32 tmp;
    int result = -1;
#if IS_SDLv1
    int ppa = (surface->flags & SDL_SRCALPHA && fmt->Amask);
#else /* IS_SDLv2 */
    int ppa;
    SDL_BlendMode mode;
    SDL_GetSurfaceBlendMode(surface, &mode);
    ppa = (fmt->Amask && mode != SDL_BLENDMODE_NONE);
#endif /* IS_SDLv2 */

#if IS_SDLv1
    pixels = (Uint8 *)surface->pixels + surface->offset +
#else  /* IS_SDLv2 */
    pixels = (Uint8 *)surface->pixels +
#endif /* IS_SDLv2 */
             (Uint16)rect->y * surface->pitch + (Uint16)rect->x * bpp;
    skip = surface->pitch - width * bpp;

    switch (bpp) {
        case 1: {
            SDL_GetRGBA(color, fmt, &cR, &cG, &cB, &cA);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, pixels, fmt);
                        BLEND_ADD(tmp, cR, cG, cB, cA, sR, sG, sB, sA);
                        *pixels = SDL_MapRGBA(fmt, sR, sG, sB, sA);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        case 3: {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24(offsetR, offsetG, offsetB, fmt);
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_ADD(tmp, cR, cG, cB, cA, sR, sG, sB, sA);
                        pixels[offsetR] = sR;
                        pixels[offsetG] = sG;
                        pixels[offsetB] = sB;
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        default: {
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            /*
            printf ("Color: %d, %d, %d, %d, BPP is: %d\n", cR, cG, cB, cA,
            bpp);
            */
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_ADD(tmp, cR, cG, cB, cA, sR, sG, sB, sA);
                        CREATE_PIXEL(pixels, sR, sG, sB, sA, bpp, fmt);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
    }
    return result;
}

static int
surface_fill_blend_sub(SDL_Surface *surface, SDL_Rect *rect, Uint32 color)
{
    Uint8 *pixels;
    int width = rect->w;
    int height = rect->h;
    int skip;
    int bpp = surface->format->BytesPerPixel;
    int n;
    SDL_PixelFormat *fmt = surface->format;
    Uint8 sR, sG, sB, sA, cR, cG, cB, cA;
    Uint32 pixel;
    Sint32 tmp2;
    int result = -1;
#if IS_SDLv1
    int ppa = (surface->flags & SDL_SRCALPHA && fmt->Amask);
#else /* IS_SDLv2 */
    int ppa;
    SDL_BlendMode mode;
    SDL_GetSurfaceBlendMode(surface, &mode);
    ppa = (fmt->Amask && mode != SDL_BLENDMODE_NONE);
#endif /* IS_SDLv2 */

#if IS_SDLv1
    pixels = (Uint8 *)surface->pixels + surface->offset +
#else  /* IS_SDLv2 */
    pixels = (Uint8 *)surface->pixels +
#endif /* IS_SDLv2 */
             (Uint16)rect->y * surface->pitch + (Uint16)rect->x * bpp;
    skip = surface->pitch - width * bpp;

    switch (bpp) {
        case 1: {
            SDL_GetRGBA(color, fmt, &cR, &cG, &cB, &cA);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, pixels, fmt);
                        BLEND_SUB(tmp2, cR, cG, cB, cA, sR, sG, sB, sA);
                        *pixels = SDL_MapRGBA(fmt, sR, sG, sB, sA);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        case 3: {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24(offsetR, offsetG, offsetB, fmt);
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_SUB(tmp2, cR, cG, cB, cA, sR, sG, sB, sA);
                        pixels[offsetR] = sR;
                        pixels[offsetG] = sG;
                        pixels[offsetB] = sB;
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        default: {
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            /*
            printf ("Color: %d, %d, %d, %d, BPP is: %d\n", cR, cG, cB, cA,
            bpp);
            */
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_SUB(tmp2, cR, cG, cB, cA, sR, sG, sB, sA);
                        CREATE_PIXEL(pixels, sR, sG, sB, sA, bpp, fmt);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
    }
    return result;
}

static int
surface_fill_blend_mult(SDL_Surface *surface, SDL_Rect *rect, Uint32 color)
{
    Uint8 *pixels;
    int width = rect->w;
    int height = rect->h;
    int skip;
    int bpp = surface->format->BytesPerPixel;
    int n;
    SDL_PixelFormat *fmt = surface->format;
    Uint8 sR, sG, sB, sA, cR, cG, cB, cA;
    Uint32 pixel;
    int result = -1;
#if IS_SDLv1
    int ppa = (surface->flags & SDL_SRCALPHA && fmt->Amask);
#else /* IS_SDLv2 */
    int ppa;
    SDL_BlendMode mode;
    SDL_GetSurfaceBlendMode(surface, &mode);
    ppa = (fmt->Amask && mode != SDL_BLENDMODE_NONE);
#endif /* IS_SDLv2 */

#if IS_SDLv1
    pixels = (Uint8 *)surface->pixels + surface->offset +
#else  /* IS_SDLv2 */
    pixels = (Uint8 *)surface->pixels +
#endif /* IS_SDLv2 */
             (Uint16)rect->y * surface->pitch + (Uint16)rect->x * bpp;
    skip = surface->pitch - width * bpp;

    switch (bpp) {
        case 1: {
            SDL_GetRGBA(color, fmt, &cR, &cG, &cB, &cA);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, pixels, fmt);
                        BLEND_MULT(cR, cG, cB, cA, sR, sG, sB, sA);
                        *pixels = SDL_MapRGBA(fmt, sR, sG, sB, sA);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        case 3: {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24(offsetR, offsetG, offsetB, fmt);
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_MULT(cR, cG, cB, cA, sR, sG, sB, sA);
                        pixels[offsetR] = sR;
                        pixels[offsetG] = sG;
                        pixels[offsetB] = sB;
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        default: {
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            /*
            printf ("Color: %d, %d, %d, %d, BPP is: %d\n", cR, cG, cB, cA,
            bpp);
            */
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_MULT(cR, cG, cB, cA, sR, sG, sB, sA);
                        CREATE_PIXEL(pixels, sR, sG, sB, sA, bpp, fmt);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
    }
    return result;
}

static int
surface_fill_blend_min(SDL_Surface *surface, SDL_Rect *rect, Uint32 color)
{
    Uint8 *pixels;
    int width = rect->w;
    int height = rect->h;
    int skip;
    int bpp = surface->format->BytesPerPixel;
    int n;
    SDL_PixelFormat *fmt = surface->format;
    Uint8 sR, sG, sB, sA, cR, cG, cB, cA;
    Uint32 pixel;
    int result = -1;
#if IS_SDLv1
    int ppa = (surface->flags & SDL_SRCALPHA && fmt->Amask);
#else /* IS_SDLv2 */
    int ppa;
    SDL_BlendMode mode;
    SDL_GetSurfaceBlendMode(surface, &mode);
    ppa = (fmt->Amask && mode != SDL_BLENDMODE_NONE);
#endif /* IS_SDLv2 */

#if IS_SDLv1
    pixels = (Uint8 *)surface->pixels + surface->offset +
#else  /* IS_SDLv2 */
    pixels = (Uint8 *)surface->pixels +
#endif /* IS_SDLv2 */
             (Uint16)rect->y * surface->pitch + (Uint16)rect->x * bpp;
    skip = surface->pitch - width * bpp;

    switch (bpp) {
        case 1: {
            SDL_GetRGBA(color, fmt, &cR, &cG, &cB, &cA);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, pixels, fmt);
                        BLEND_MIN(cR, cG, cB, cA, sR, sG, sB, sA);
                        *pixels = SDL_MapRGBA(fmt, sR, sG, sB, sA);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        case 3: {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24(offsetR, offsetG, offsetB, fmt);
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_MIN(cR, cG, cB, cA, sR, sG, sB, sA);
                        pixels[offsetR] = sR;
                        pixels[offsetG] = sG;
                        pixels[offsetB] = sB;
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        default: {
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            /*
            printf ("Color: %d, %d, %d, %d, BPP is: %d\n", cR, cG, cB, cA,
            bpp);
            */
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_MIN(cR, cG, cB, cA, sR, sG, sB, sA);
                        CREATE_PIXEL(pixels, sR, sG, sB, sA, bpp, fmt);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
    }
    return result;
}

static int
surface_fill_blend_max(SDL_Surface *surface, SDL_Rect *rect, Uint32 color)
{
    Uint8 *pixels;
    int width = rect->w;
    int height = rect->h;
    int skip;
    int bpp = surface->format->BytesPerPixel;
    int n;
    SDL_PixelFormat *fmt = surface->format;
    Uint8 sR, sG, sB, sA, cR, cG, cB, cA;
    Uint32 pixel;
    int result = -1;
#if IS_SDLv1
    int ppa = (surface->flags & SDL_SRCALPHA && fmt->Amask);
#else /* IS_SDLv2 */
    int ppa;
    SDL_BlendMode mode;
    SDL_GetSurfaceBlendMode(surface, &mode);
    ppa = (fmt->Amask && mode != SDL_BLENDMODE_NONE);
#endif /* IS_SDLv2 */

#if IS_SDLv1
    pixels = (Uint8 *)surface->pixels + surface->offset +
#else  /* IS_SDLv2 */
    pixels = (Uint8 *)surface->pixels +
#endif /* IS_SDLv2 */
             (Uint16)rect->y * surface->pitch + (Uint16)rect->x * bpp;
    skip = surface->pitch - width * bpp;

    switch (bpp) {
        case 1: {
            SDL_GetRGBA(color, fmt, &cR, &cG, &cB, &cA);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, pixels, fmt);
                        BLEND_MAX(cR, cG, cB, cA, sR, sG, sB, sA);
                        *pixels = SDL_MapRGBA(fmt, sR, sG, sB, sA);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        case 3: {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24(offsetR, offsetG, offsetB, fmt);
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_MAX(cR, cG, cB, cA, sR, sG, sB, sA);
                        pixels[offsetR] = sR;
                        pixels[offsetG] = sG;
                        pixels[offsetB] = sB;
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        default: {
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            /*
            printf ("Color: %d, %d, %d, %d, BPP is: %d\n", cR, cG, cB, cA,
            bpp);
            */
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_MAX(cR, cG, cB, cA, sR, sG, sB, sA);
                        CREATE_PIXEL(pixels, sR, sG, sB, sA, bpp, fmt);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
    }
    return result;
}

/* ------------------------- */

static int
surface_fill_blend_rgba_add(SDL_Surface *surface, SDL_Rect *rect, Uint32 color)
{
    Uint8 *pixels;
    int width = rect->w;
    int height = rect->h;
    int skip;
    int bpp = surface->format->BytesPerPixel;
    int n;
    SDL_PixelFormat *fmt = surface->format;
    Uint8 sR, sG, sB, sA, cR, cG, cB, cA;
    Uint32 pixel;
    Uint32 tmp;
    int result = -1;
#if IS_SDLv1
    int ppa = (surface->flags & SDL_SRCALPHA && fmt->Amask);
#else /* IS_SDLv2 */
    int ppa;
    SDL_BlendMode mode;
    SDL_GetSurfaceBlendMode(surface, &mode);
    ppa = (fmt->Amask && mode != SDL_BLENDMODE_NONE);
#endif /* IS_SDLv2 */

    if (!ppa) {
        return surface_fill_blend_add(surface, rect, color);
    }

#if IS_SDLv1
    pixels = (Uint8 *)surface->pixels + surface->offset +
#else  /* IS_SDLv2 */
    pixels = (Uint8 *)surface->pixels +
#endif /* IS_SDLv2 */
             (Uint16)rect->y * surface->pitch + (Uint16)rect->x * bpp;
    skip = surface->pitch - width * bpp;

    switch (bpp) {
        case 1: {
            SDL_GetRGBA(color, fmt, &cR, &cG, &cB, &cA);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, pixels, fmt);
                        BLEND_RGBA_ADD(tmp, cR, cG, cB, cA, sR, sG, sB, sA);
                        *pixels = SDL_MapRGBA(fmt, sR, sG, sB, sA);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        default: {
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            /*
            printf ("Color: %d, %d, %d, %d, BPP is: %d\n", cR, cG, cB, cA,
            bpp);
            */
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_RGBA_ADD(tmp, cR, cG, cB, cA, sR, sG, sB, sA);
                        CREATE_PIXEL(pixels, sR, sG, sB, sA, bpp, fmt);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
    }
    return result;
}

static int
surface_fill_blend_rgba_sub(SDL_Surface *surface, SDL_Rect *rect, Uint32 color)
{
    Uint8 *pixels;
    int width = rect->w;
    int height = rect->h;
    int skip;
    int bpp = surface->format->BytesPerPixel;
    int n;
    SDL_PixelFormat *fmt = surface->format;
    Uint8 sR, sG, sB, sA, cR, cG, cB, cA;
    Uint32 pixel;
    Sint32 tmp2;
    int result = -1;
#if IS_SDLv1
    int ppa = (surface->flags & SDL_SRCALPHA && fmt->Amask);
#else /* IS_SDLv2 */
    int ppa;
    SDL_BlendMode mode;
    SDL_GetSurfaceBlendMode(surface, &mode);
    ppa = (fmt->Amask && mode != SDL_BLENDMODE_NONE);
#endif /* IS_SDLv2 */

    if (!ppa) {
        return surface_fill_blend_sub(surface, rect, color);
    }

#if IS_SDLv1
    pixels = (Uint8 *)surface->pixels + surface->offset +
#else  /* IS_SDLv2 */
    pixels = (Uint8 *)surface->pixels +
#endif /* IS_SDLv2 */
             (Uint16)rect->y * surface->pitch + (Uint16)rect->x * bpp;
    skip = surface->pitch - width * bpp;

    switch (bpp) {
        case 1: {
            SDL_GetRGBA(color, fmt, &cR, &cG, &cB, &cA);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, pixels, fmt);
                        BLEND_RGBA_SUB(tmp2, cR, cG, cB, cA, sR, sG, sB, sA);
                        *pixels = SDL_MapRGBA(fmt, sR, sG, sB, sA);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        default: {
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            /*
            printf ("Color: %d, %d, %d, %d, BPP is: %d\n", cR, cG, cB, cA,
            bpp);
            */
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_RGBA_SUB(tmp2, cR, cG, cB, cA, sR, sG, sB, sA);
                        CREATE_PIXEL(pixels, sR, sG, sB, sA, bpp, fmt);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
    }
    return result;
}

static int
surface_fill_blend_rgba_mult(SDL_Surface *surface, SDL_Rect *rect,
                             Uint32 color)
{
    Uint8 *pixels;
    int width = rect->w;
    int height = rect->h;
    int skip;
    int bpp = surface->format->BytesPerPixel;
    int n;
    SDL_PixelFormat *fmt = surface->format;
    Uint8 sR, sG, sB, sA, cR, cG, cB, cA;
    Uint32 pixel;
    int result = -1;
#if IS_SDLv1
    int ppa = (surface->flags & SDL_SRCALPHA && fmt->Amask);
#else /* IS_SDLv2 */
    int ppa;
    SDL_BlendMode mode;
    SDL_GetSurfaceBlendMode(surface, &mode);
    ppa = (fmt->Amask && mode != SDL_BLENDMODE_NONE);
#endif /* IS_SDLv2 */

    if (!ppa) {
        return surface_fill_blend_mult(surface, rect, color);
    }

#if IS_SDLv1
    pixels = (Uint8 *)surface->pixels + surface->offset +
#else  /* IS_SDLv2 */
    pixels = (Uint8 *)surface->pixels +
#endif /* IS_SDLv2 */
             (Uint16)rect->y * surface->pitch + (Uint16)rect->x * bpp;
    skip = surface->pitch - width * bpp;

    switch (bpp) {
        case 1: {
            SDL_GetRGBA(color, fmt, &cR, &cG, &cB, &cA);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, pixels, fmt);
                        BLEND_RGBA_MULT(cR, cG, cB, cA, sR, sG, sB, sA);
                        *pixels = SDL_MapRGBA(fmt, sR, sG, sB, sA);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        default: {
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            /*
            printf ("Color: %d, %d, %d, %d, BPP is: %d\n", cR, cG, cB, cA,
            bpp);
            */
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_RGBA_MULT(cR, cG, cB, cA, sR, sG, sB, sA);
                        CREATE_PIXEL(pixels, sR, sG, sB, sA, bpp, fmt);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
    }
    return result;
}

static int
surface_fill_blend_rgba_min(SDL_Surface *surface, SDL_Rect *rect, Uint32 color)
{
    Uint8 *pixels;
    int width = rect->w;
    int height = rect->h;
    int skip;
    int bpp = surface->format->BytesPerPixel;
    int n;
    SDL_PixelFormat *fmt = surface->format;
    Uint8 sR, sG, sB, sA, cR, cG, cB, cA;
    Uint32 pixel;
    int result = -1;
#if IS_SDLv1
    int ppa = (surface->flags & SDL_SRCALPHA && fmt->Amask);
#else /* IS_SDLv2 */
    int ppa;
    SDL_BlendMode mode;
    SDL_GetSurfaceBlendMode(surface, &mode);
    ppa = (fmt->Amask && mode != SDL_BLENDMODE_NONE);
#endif /* IS_SDLv2 */

    if (!ppa) {
        return surface_fill_blend_min(surface, rect, color);
    }

#if IS_SDLv1
    pixels = (Uint8 *)surface->pixels + surface->offset +
#else  /* IS_SDLv2 */
    pixels = (Uint8 *)surface->pixels +
#endif /* IS_SDLv2 */
             (Uint16)rect->y * surface->pitch + (Uint16)rect->x * bpp;
    skip = surface->pitch - width * bpp;

    switch (bpp) {
        case 1: {
            SDL_GetRGBA(color, fmt, &cR, &cG, &cB, &cA);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, pixels, fmt);
                        BLEND_RGBA_MIN(cR, cG, cB, cA, sR, sG, sB, sA);
                        *pixels = SDL_MapRGBA(fmt, sR, sG, sB, sA);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        default: {
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            /*
            printf ("Color: %d, %d, %d, %d, BPP is: %d\n", cR, cG, cB, cA,
            bpp);
            */
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_RGBA_MIN(cR, cG, cB, cA, sR, sG, sB, sA);
                        CREATE_PIXEL(pixels, sR, sG, sB, sA, bpp, fmt);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
    }
    return result;
}

static int
surface_fill_blend_rgba_max(SDL_Surface *surface, SDL_Rect *rect, Uint32 color)
{
    Uint8 *pixels;
    int width = rect->w;
    int height = rect->h;
    int skip;
    int bpp = surface->format->BytesPerPixel;
    int n;
    SDL_PixelFormat *fmt = surface->format;
    Uint8 sR, sG, sB, sA, cR, cG, cB, cA;
    Uint32 pixel;
    int result = -1;
#if IS_SDLv1
    int ppa = (surface->flags & SDL_SRCALPHA && fmt->Amask);
#else /* IS_SDLv2 */
    int ppa;
    SDL_BlendMode mode;
    SDL_GetSurfaceBlendMode(surface, &mode);
    ppa = (fmt->Amask && mode != SDL_BLENDMODE_NONE);
#endif /* IS_SDLv2 */

    if (!ppa) {
        return surface_fill_blend_max(surface, rect, color);
    }

#if IS_SDLv1
    pixels = (Uint8 *)surface->pixels + surface->offset +
#else  /* IS_SDLv2 */
    pixels = (Uint8 *)surface->pixels +
#endif /* IS_SDLv2 */
             (Uint16)rect->y * surface->pitch + (Uint16)rect->x * bpp;
    skip = surface->pitch - width * bpp;

    switch (bpp) {
        case 1: {
            SDL_GetRGBA(color, fmt, &cR, &cG, &cB, &cA);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, pixels, fmt);
                        BLEND_RGBA_MAX(cR, cG, cB, cA, sR, sG, sB, sA);
                        *pixels = SDL_MapRGBA(fmt, sR, sG, sB, sA);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
        default: {
            GET_PIXELVALS(cR, cG, cB, cA, color, fmt, ppa);
            /*
            printf ("Color: %d, %d, %d, %d, BPP is: %d\n", cR, cG, cB, cA,
            bpp);
            */
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, bpp, pixels);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, fmt, ppa);
                        BLEND_RGBA_MAX(cR, cG, cB, cA, sR, sG, sB, sA);
                        CREATE_PIXEL(pixels, sR, sG, sB, sA, bpp, fmt);
                        pixels += bpp;
                    },
                    n, width);
                pixels += skip;
            }
            result = 0;
            break;
        }
    }
    return result;
}

int
surface_fill_blend(SDL_Surface *surface, SDL_Rect *rect, Uint32 color,
                   int blendargs)
{
    int result = -1;
    int locked = 0;

    surface_respect_clip_rect(surface, rect);

    /* Lock the surface, if needed */
    if (SDL_MUSTLOCK(surface)) {
        if (SDL_LockSurface(surface) < 0)
            return -1;
        locked = 1;
    }

    switch (blendargs) {
        case PYGAME_BLEND_ADD: {
            result = surface_fill_blend_add(surface, rect, color);
            break;
        }
        case PYGAME_BLEND_SUB: {
            result = surface_fill_blend_sub(surface, rect, color);
            break;
        }
        case PYGAME_BLEND_MULT: {
            result = surface_fill_blend_mult(surface, rect, color);
            break;
        }
        case PYGAME_BLEND_MIN: {
            result = surface_fill_blend_min(surface, rect, color);
            break;
        }
        case PYGAME_BLEND_MAX: {
            result = surface_fill_blend_max(surface, rect, color);
            break;
        }

        case PYGAME_BLEND_RGBA_ADD: {
            result = surface_fill_blend_rgba_add(surface, rect, color);
            break;
        }
        case PYGAME_BLEND_RGBA_SUB: {
            result = surface_fill_blend_rgba_sub(surface, rect, color);
            break;
        }
        case PYGAME_BLEND_RGBA_MULT: {
            result = surface_fill_blend_rgba_mult(surface, rect, color);
            break;
        }
        case PYGAME_BLEND_RGBA_MIN: {
            result = surface_fill_blend_rgba_min(surface, rect, color);
            break;
        }
        case PYGAME_BLEND_RGBA_MAX: {
            result = surface_fill_blend_rgba_max(surface, rect, color);
            break;
        }

        default: {
            result = -1;
            break;
        }
    }

    if (locked) {
        SDL_UnlockSurface(surface);
    }
    return result;
}
