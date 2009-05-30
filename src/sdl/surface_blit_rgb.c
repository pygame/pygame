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

void
blit_blend_add (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
    Uint32          tmp;

    if (srcbpp == 4 && dstbpp == 4)
    {
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA);
                GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA);
                D_BLEND_RGB_ADD (tmp, sR, sG, sB, dR, dG, dB);
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                src += srcbpp;
                dst += dstbpp;
            }, n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_ADD (tmp, sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_ADD (tmp, sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_ADD (tmp, sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_ADD (tmp, sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    /*
                    printf ("VAL: %d %d %d || %d %d %d || 0x%.8x || 0x%.8x \n",
                        sR, sG, sB, dR, dG, dB, *((Uint16*)dst),
                        ((Uint16)SDL_MapRGB(dstfmt, dR, dG, dB)));
                    */
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

void
blit_blend_sub (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
    Sint32          tmp2;

    if (srcbpp == 4 && dstbpp == 4)
    {
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA);
                GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA);
                D_BLEND_RGB_SUB (tmp2, sR, sG, sB, dR, dG, dB);
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                src += srcbpp;
                dst += dstbpp;
            }, n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_SUB (tmp2, sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_SUB (tmp2, sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_SUB (tmp2, sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_SUB (tmp2, sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

void
blit_blend_mul (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;

    if (srcfmt->BytesPerPixel == 4 && dstfmt->BytesPerPixel == 4)
    {
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA);
                GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA);
                D_BLEND_RGB_MULT (sR, sG, sB, dR, dG, dB);
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                src += srcbpp;
                dst += dstbpp;
            }, n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_MULT (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_MULT (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_MULT (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_MULT (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

void
blit_blend_min (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;

    if (srcfmt->BytesPerPixel == 4 && dstfmt->BytesPerPixel == 4)
    {
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA);
                GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA);
                D_BLEND_RGB_MIN (sR, sG, sB, dR, dG, dB);
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                src += srcbpp;
                dst += dstbpp;
            }, n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_MIN (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_MIN (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_MIN (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_MIN (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

void
blit_blend_max (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;

    if (srcfmt->BytesPerPixel == 4 && dstfmt->BytesPerPixel == 4)
    {
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA);
                GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA);
                D_BLEND_RGB_MAX (sR, sG, sB, dR, dG, dB);
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                src += srcbpp;
                dst += dstbpp;
            }, n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }


    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_MAX (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_MAX (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_MAX (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_MAX (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

void
blit_blend_xor (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;

    if (srcbpp == 4 && dstbpp == 4)
    {
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA);
                GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA);
                D_BLEND_RGB_XOR (sR, sG, sB, dR, dG, dB);
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                src += srcbpp;
                dst += dstbpp;
            }, n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }


    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_XOR (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_XOR (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_XOR (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_XOR (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

void
blit_blend_and (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;

    if (srcbpp == 4 && dstbpp == 4)
    {
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA);
                GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA);
                D_BLEND_RGB_AND (sR, sG, sB, dR, dG, dB);
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                src += srcbpp;
                dst += dstbpp;
            }, n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_AND (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_AND (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_AND (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_AND (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

void
blit_blend_or (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;

    if (srcbpp == 4 && dstbpp == 4)
    {
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA);
                GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA);
                D_BLEND_RGB_OR (sR, sG, sB, dR, dG, dB);
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                src += srcbpp;
                dst += dstbpp;
            }, n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_OR (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_OR (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_OR (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_OR (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

void
blit_blend_diff (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;

    if (srcbpp == 4 && dstbpp == 4)
    {
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA);
                GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA);
                D_BLEND_RGB_DIFF (sR, sG, sB, dR, dG, dB);
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                src += srcbpp;
                dst += dstbpp;
            }, n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_DIFF (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_DIFF (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_DIFF (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_DIFF (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

void
blit_blend_screen (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;

    if (srcbpp == 4 && dstbpp == 4)
    {
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA);
                GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA);
                D_BLEND_RGB_SCREEN (sR, sG, sB, dR, dG, dB);
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                src += srcbpp;
                dst += dstbpp;
            }, n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_SCREEN (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_SCREEN (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_SCREEN (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_SCREEN (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

void
blit_blend_avg (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;

    if (srcbpp == 4 && dstbpp == 4)
    {
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                GET_RGB_VALS ((*(Uint32*)src), srcfmt, sR, sG, sB, sA);
                GET_RGB_VALS ((*(Uint32*)dst), dstfmt, dR, dG, dB, dA);
                D_BLEND_RGB_AVG (sR, sG, sB, dR, dG, dB);
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                src += srcbpp;
                dst += dstbpp;
            }, n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_AVG (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_AVG (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_AVG (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    D_BLEND_RGB_AVG (sR, sG, sB, dR, dG, dB);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}
