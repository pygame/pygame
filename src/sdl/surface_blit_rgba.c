/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners, 2006 Rene Dudfield,
                2007-2009 Marcus von Appen

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

#include "surface_blit.h"
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef HAVE_OPENMP
#define CREATE_BLITTER(_name,_blitop)                                   \
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
        Uint32          tmp;                                            \
        Sint32          tmp2;                                           \
        Uint8          *sppx, *dppx;                                    \
        int             x, y;                                           \
                                                                        \
        if (srcbpp == 4 && dstbpp == 4)                                 \
        {                                                               \
            _Pragma("omp parallel")                                     \
            {                                                           \
                _Pragma("omp for private(sppx,dppx,x,sR,sG,sB,sA,dR,dG,dB,dA,tmp,tmp2)") \
                for (y = 0; y < height; y++)                            \
                {                                                       \
                    for (x = 0; x < width; x++)                         \
                    {                                                   \
                        sppx = src + y * info->s_pitch + x * srcbpp;    \
                        dppx = dst + y * info->d_pitch + x * dstbpp;    \
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
                _Pragma("omp parallel")                                 \
                {                                                       \
                    _Pragma("omp for private(sppx,dppx,x,sR,sG,sB,sA,dR,dG,dB,dA,tmp,tmp2)") \
                    for (y = 0; y < height; y++)                        \
                    {                                                   \
                        for (x = 0; x < width; x++)                     \
                        {                                               \
                            sppx = src + y * info->s_pitch + x * srcbpp; \
                            dppx = dst + y * info->d_pitch + x * dstbpp; \
                            GET_PALETTE_VALS(sppx, srcfmt, sR, sG, sB, sA); \
                            GET_PALETTE_VALS(dppx, dstfmt, dR, dG, dB, dA); \
                            _blitop;                                    \
                            CREATE_PIXEL(dppx, dR, dG, dB, dA, dstbpp, dstfmt); \
                        }                                               \
                    }                                                   \
                }                                                       \
            }                                                           \
            else /* dstbpp > 1 */                                       \
            {                                                           \
                _Pragma("omp parallel")                                 \
                {                                                       \
                    _Pragma("omp for private(sppx,dppx,pixel,x,sR,sG,sB,sA,dR,dG,dB,dA,tmp,tmp2)") \
                    for (y = 0; y < height; y++)                        \
                    {                                                   \
                        for (x = 0; x < width; x++)                     \
                        {                                               \
                            sppx = src + y * info->s_pitch + x * srcbpp; \
                            dppx = dst + y * info->d_pitch + x * dstbpp; \
                            GET_PALETTE_VALS(sppx, srcfmt, sR, sG, sB, sA); \
                            GET_PIXEL (pixel, dstbpp, dppx);             \
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
                _Pragma("omp parallel")                                 \
                {                                                       \
                    _Pragma("omp for private(sppx,dppx,pixel,x,sR,sG,sB,sA,dR,dG,dB,dA,tmp,tmp2)") \
                    for (y = 0; y < height; y++)                        \
                    {                                                   \
                        for (x = 0; x < width; x++)                     \
                        {                                               \
                            sppx = src + y * info->s_pitch + x * srcbpp; \
                            dppx = dst + y * info->d_pitch + x * dstbpp; \
                            GET_PIXEL(pixel, srcbpp, sppx);             \
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
                _Pragma("omp parallel")                                 \
                {                                                       \
                    _Pragma("omp for private(sppx,dppx,pixel,x,sR,sG,sB,sA,dR,dG,dB,dA,tmp,tmp2)") \
                    for (y = 0; y < height; y++)                        \
                    {                                                   \
                        for (x = 0; x < width; x++)                     \
                        {                                               \
                            sppx = src + y * info->s_pitch + x * srcbpp; \
                            dppx = dst + y * info->d_pitch + x * dstbpp; \
                            GET_PIXEL(pixel, srcbpp, sppx);             \
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
#define CREATE_BLITTER(_name,_blitop)                                   \
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
        Uint32          tmp;                                            \
        Sint32          tmp2;                                           \
                                                                        \
        if (srcbpp == 4 && dstbpp == 4)                                 \
        {                                                               \
            while (height--)                                            \
            {                                                           \
                LOOP_UNROLLED4(                                         \
                {                                                           \
                    GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA); \
                    GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA); \
                    _blitop;                                            \
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);      \
                    src += srcbpp;                                          \
                    dst += dstbpp;                                          \
                }, n, width);                                               \
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

CREATE_BLITTER(blend_rgba_add, D_BLEND_RGBA_ADD(tmp,sR,sG,sB,sA,dR,dG,dB,dA))
CREATE_BLITTER(blend_rgba_sub, D_BLEND_RGBA_SUB(tmp2,sR,sG,sB,sA,dR,dG,dB,dA))
CREATE_BLITTER(blend_rgba_mul, D_BLEND_RGBA_MULT(sR,sG,sB,sA,dR,dG,dB,dA))
CREATE_BLITTER(blend_rgba_min, D_BLEND_RGBA_MIN(sR,sG,sB,sA,dR,dG,dB,dA))
CREATE_BLITTER(blend_rgba_max, D_BLEND_RGBA_MAX(sR,sG,sB,sA,dR,dG,dB,dA))
CREATE_BLITTER(blend_rgba_xor, D_BLEND_RGBA_XOR(sR,sG,sB,sA,dR,dG,dB,dA))
CREATE_BLITTER(blend_rgba_and, D_BLEND_RGBA_AND(sR,sG,sB,sA,dR,dG,dB,dA))
CREATE_BLITTER(blend_rgba_or, D_BLEND_RGBA_OR(sR,sG,sB,sA,dR,dG,dB,dA))
CREATE_BLITTER(blend_rgba_diff, D_BLEND_RGBA_DIFF(sR,sG,sB,sA,dR,dG,dB,dA))
CREATE_BLITTER(blend_rgba_screen, D_BLEND_RGBA_SCREEN(sR,sG,sB,sA,dR,dG,dB,dA))
CREATE_BLITTER(blend_rgba_avg, D_BLEND_RGBA_AVG(sR,sG,sB,sA,dR,dG,dB,dA))
