/*
  pygame - Python Game Library
  Copyright (C) 2009 Marcus von Appen

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

#ifndef _PYGAME_SURFACEBLIT_H_
#define _PYGAME_SURFACEBLIT_H_

#include "surface.h"
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

/* The structure passed to the low level blit functions */
typedef struct
{
    Uint8          *s_pixels;
    int             s_width;
    int             s_height;
    int             s_skip;
    int             s_pitch;
    Uint8          *d_pixels;
    int             d_width;
    int             d_height;
    int             d_skip;
    int             d_pitch;
    void           *aux_data;
    SDL_PixelFormat *src;
    Uint8          *table;
    SDL_PixelFormat *dst;
} SDL_BlitInfo;

/* Blitter definition
 *
 * _sop1 == special alpha blit operation for 1bpp src
 * _sop2 == special alpha blit operation for >1bpp src
 * _sop3 == special alpha blit operation for 4bpp<->4bpp
 *
 */
#ifdef HAVE_OPENMP
#define CREATE_BLITTER(_name,_blitop,_sop1,_sop2,_sop3)                 \
    void blit_##_name(SDL_BlitInfo *info)                               \
    {                                                                   \
        int             width = info->d_width;                          \
        int             height = info->d_height;                        \
        Uint8          *src = info->s_pixels;                           \
        Uint8          *dst = info->d_pixels;                           \
        SDL_PixelFormat *srcfmt = info->src;                            \
        SDL_PixelFormat *dstfmt = info->dst;                            \
        int             srcbpp = srcfmt->BytesPerPixel;                 \
        int             dstbpp = dstfmt->BytesPerPixel;                 \
        Uint8           dR, dG, dB, dA, sR, sG, sB, sA;                 \
        Uint32          pixel;                                          \
        int             alpha = srcfmt->alpha;                          \
        Uint32          colorkey = srcfmt->colorkey;                    \
        Uint32          tmp;                                            \
        Sint32          tmp2;                                           \
        Uint8          *sppx, *dppx;                                    \
        int             x, y;                                           \
                                                                        \
        if (srcbpp == 4 && dstbpp == 4)                                 \
        {                                                               \
            PRAGMA(omp parallel)                                        \
            {                                                           \
                PRAGMA(omp for private(sppx,dppx,x,sR,sG,sB,sA,dR,dG,dB,dA,tmp,tmp2)) \
                for (y = 0; y < height; y++)                            \
                {                                                       \
                    for (x = 0; x < width; x++)                         \
                    {                                                   \
                        sppx = src + y * info->s_pitch + x * srcbpp;    \
                        dppx = dst + y * info->d_pitch + x * dstbpp;    \
                        _sop3;                                          \
                        GET_RGB_VALS ((*(Uint32*)sppx), srcfmt, sR, sG, sB, sA); \
                        GET_RGB_VALS ((*(Uint32*)dppx), dstfmt, dR, dG, dB, dA); \
                        _blitop;                                        \
                        CREATE_PIXEL(dppx, dR, dG, dB, dA, dstbpp, dstfmt); \
                    }                                                   \
                }                                                       \
            }                                                           \
            return;                                                     \
        }                                                               \
                                                                        \
        if (srcbpp == 1)                                                \
        {                                                               \
            if (dstbpp == 1)                                            \
            {                                                           \
                PRAGMA(omp parallel)                                    \
                {                                                       \
                    PRAGMA(omp for private(sppx,dppx,x,sR,sG,sB,sA,dR,dG,dB,dA,tmp,tmp2)) \
                    for (y = 0; y < height; y++)                        \
                    {                                                   \
                        for (x = 0; x < width; x++)                     \
                        {                                               \
                            sppx = src + y * info->s_pitch + x * srcbpp; \
                            dppx = dst + y * info->d_pitch + x * dstbpp; \
                            GET_PALETTE_VALS(sppx, srcfmt, sR, sG, sB, sA); \
                            _sop1;                                      \
                            GET_PALETTE_VALS(dppx, dstfmt, dR, dG, dB, dA); \
                            _blitop;                                    \
                            CREATE_PIXEL(dppx, dR, dG, dB, dA, dstbpp, dstfmt); \
                        }                                               \
                    }                                                   \
                }                                                       \
            }                                                           \
            else /* dstbpp > 1 */                                       \
            {                                                           \
                PRAGMA(omp parallel)                                    \
                {                                                       \
                    PRAGMA(omp for private(sppx,dppx,pixel,x,sR,sG,sB,sA,dR,dG,dB,dA,tmp,tmp2)) \
                    for (y = 0; y < height; y++)                        \
                    {                                                   \
                        for (x = 0; x < width; x++)                     \
                        {                                               \
                            sppx = src + y * info->s_pitch + x * srcbpp; \
                            dppx = dst + y * info->d_pitch + x * dstbpp; \
                            GET_PALETTE_VALS(sppx, srcfmt, sR, sG, sB, sA); \
                            _sop1;                                      \
                            GET_PIXEL (pixel, dstbpp, dppx);            \
                            GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA); \
                            _blitop;                                    \
                            CREATE_PIXEL(dppx, dR, dG, dB, dA, dstbpp, dstfmt); \
                        }                                               \
                    }                                                   \
                }                                                       \
            }                                                           \
        }                                                               \
        else /* srcbpp > 1 */                                           \
        {                                                               \
            if (dstbpp == 1)                                            \
            {                                                           \
                PRAGMA(omp parallel)                                    \
                {                                                       \
                    PRAGMA(omp for private(sppx,dppx,pixel,x,sR,sG,sB,sA,dR,dG,dB,dA,tmp,tmp2)) \
                    for (y = 0; y < height; y++)                        \
                    {                                                   \
                        for (x = 0; x < width; x++)                     \
                        {                                               \
                            sppx = src + y * info->s_pitch + x * srcbpp; \
                            dppx = dst + y * info->d_pitch + x * dstbpp; \
                            GET_PIXEL(pixel, srcbpp, sppx);             \
                            _sop2;                                      \
                            GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA); \
                            GET_PALETTE_VALS(dppx, dstfmt, dR, dG, dB, dA); \
                            _blitop;                                    \
                            CREATE_PIXEL(dppx, dR, dG, dB, dA, dstbpp, dstfmt); \
                        }                                               \
                    }                                                   \
                }                                                       \
            }                                                           \
            else /* dstbpp > 1 */                                       \
            {                                                           \
                PRAGMA(omp parallel)                                    \
                {                                                       \
                    PRAGMA(omp for private(sppx,dppx,pixel,x,sR,sG,sB,sA,dR,dG,dB,dA,tmp,tmp2)) \
                    for (y = 0; y < height; y++)                        \
                    {                                                   \
                        for (x = 0; x < width; x++)                     \
                        {                                               \
                            sppx = src + y * info->s_pitch + x * srcbpp; \
                            dppx = dst + y * info->d_pitch + x * dstbpp; \
                            GET_PIXEL(pixel, srcbpp, sppx);             \
                            _sop2;                                      \
                            GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA); \
                            GET_PIXEL (pixel, dstbpp, dppx);            \
                            GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA); \
                            _blitop;                                    \
                            CREATE_PIXEL(dppx, dR, dG, dB, dA, dstbpp, dstfmt); \
                        }                                               \
                    }                                                   \
                }                                                       \
            }                                                           \
        }                                                               \
    }
#else /* HAVE_OPENMP */
#define CREATE_BLITTER(_name,_blitop,_sop1,_sop2,_sop3)                 \
    void blit_##_name(SDL_BlitInfo *info)                               \
    {                                                                   \
        int             n;                                              \
        int             width = info->d_width;                          \
        int             height = info->d_height;                        \
        Uint8          *src = info->s_pixels;                           \
        int             srcskip = info->s_skip;                         \
        Uint8          *dst = info->d_pixels;                           \
        int             dstskip = info->d_skip;                         \
        SDL_PixelFormat *srcfmt = info->src;                            \
        SDL_PixelFormat *dstfmt = info->dst;                            \
        int             srcbpp = srcfmt->BytesPerPixel;                 \
        int             dstbpp = dstfmt->BytesPerPixel;                 \
        Uint8           dR, dG, dB, dA, sR, sG, sB, sA;                 \
        Uint32          pixel;                                          \
        int             alpha = srcfmt->alpha;                          \
        Uint32          colorkey = srcfmt->colorkey;                    \
        Uint32          tmp;                                            \
        Sint32          tmp2;                                           \
                                                                        \
        if (srcbpp == 4 && dstbpp == 4)                                 \
        {                                                               \
            while (height--)                                            \
            {                                                           \
                LOOP_UNROLLED4(                                         \
                {                                                       \
                    GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA); \
                    GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA); \
                    _sop3;                                              \
                    _blitop;                                            \
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);  \
                    src += srcbpp;                                      \
                    dst += dstbpp;                                      \
                }, n, width);                                           \
                src += srcskip;                                         \
                dst += dstskip;                                         \
            }                                                           \
            return;                                                     \
        }                                                               \
                                                                        \
        if (srcbpp == 1)                                                \
        {                                                               \
            if (dstbpp == 1)                                            \
            {                                                           \
                while (height--)                                        \
                {                                                       \
                    LOOP_UNROLLED4(                                     \
                    {                                                   \
                        GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);  \
                        _sop1;                                          \
                        GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);  \
                        _blitop;                                        \
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt); \
                        src += srcbpp;                                  \
                        dst += dstbpp;                                  \
                    }, n, width);                                       \
                    src += srcskip;                                     \
                    dst += dstskip;                                     \
                }                                                       \
            }                                                           \
            else /* dstbpp > 1 */                                       \
            {                                                           \
                while (height--)                                        \
                {                                                       \
                    LOOP_UNROLLED4(                                     \
                    {                                                   \
                        GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);  \
                        _sop1;                                          \
                        GET_PIXEL (pixel, dstbpp, dst);                 \
                        GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);   \
                        _blitop;                                        \
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt); \
                        src += srcbpp;                                  \
                        dst += dstbpp;                                  \
                    }, n, width);                                       \
                    src += srcskip;                                     \
                    dst += dstskip;                                     \
                }                                                       \
            }                                                           \
        }                                                               \
        else /* srcbpp > 1 */                                           \
        {                                                               \
            if (dstbpp == 1)                                            \
            {                                                           \
                while (height--)                                        \
                {                                                       \
                    LOOP_UNROLLED4(                                     \
                    {                                                   \
                        GET_PIXEL(pixel, srcbpp, src);                  \
                        _sop2;                                          \
                        GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);   \
                        GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);  \
                        _blitop;                                        \
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt); \
                        src += srcbpp;                                  \
                        dst += dstbpp;                                  \
                    }, n, width);                                       \
                    src += srcskip;                                     \
                    dst += dstskip;                                     \
                }                                                       \
                                                                        \
            }                                                           \
            else /* dstbpp > 1 */                                       \
            {                                                           \
                while (height--)                                        \
                {                                                       \
                    LOOP_UNROLLED4(                                     \
                    {                                                   \
                        GET_PIXEL(pixel, srcbpp, src);                  \
                        _sop2;                                          \
                        GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);   \
                        GET_PIXEL (pixel, dstbpp, dst);                 \
                        GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);   \
                        _blitop;                                        \
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt); \
                        src += srcbpp;                                  \
                        dst += dstbpp;                                  \
                    }, n, width);                                       \
                    src += srcskip;                                     \
                    dst += dstskip;                                     \
                }                                                       \
            }                                                           \
        }                                                               \
    }
#endif /* HAVE_OPENMP */

void blit_blend_rgb_add (SDL_BlitInfo* info);
void blit_blend_rgb_sub (SDL_BlitInfo* info);
void blit_blend_rgb_mul (SDL_BlitInfo* info);
void blit_blend_rgb_min (SDL_BlitInfo* info);
void blit_blend_rgb_max (SDL_BlitInfo* info);
void blit_blend_rgb_and (SDL_BlitInfo* info);
void blit_blend_rgb_or (SDL_BlitInfo* info);
void blit_blend_rgb_xor (SDL_BlitInfo* info);
void blit_blend_rgb_diff (SDL_BlitInfo* info);
void blit_blend_rgb_screen (SDL_BlitInfo* info);
void blit_blend_rgb_avg (SDL_BlitInfo* info);

void blit_blend_rgba_add (SDL_BlitInfo* info);
void blit_blend_rgba_sub (SDL_BlitInfo* info);
void blit_blend_rgba_mul (SDL_BlitInfo* info);
void blit_blend_rgba_min (SDL_BlitInfo* info);
void blit_blend_rgba_max (SDL_BlitInfo* info);
void blit_blend_rgba_and (SDL_BlitInfo* info);
void blit_blend_rgba_or (SDL_BlitInfo* info);
void blit_blend_rgba_xor (SDL_BlitInfo* info);
void blit_blend_rgba_diff (SDL_BlitInfo* info);
void blit_blend_rgba_screen (SDL_BlitInfo* info);
void blit_blend_rgba_avg (SDL_BlitInfo* info);

void blit_alpha_alpha (SDL_BlitInfo* info);
void blit_alpha_colorkey (SDL_BlitInfo* info);
void blit_alpha_solid (SDL_BlitInfo* info);

#endif /* _PYGAME_SURFACEBLIT_H_ */
