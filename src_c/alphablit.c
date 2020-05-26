/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners
  Copyright (C) 2006 Rene Dudfield
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

  Pete Shinners
  pete@shinners.org
*/

#define NO_PYGAME_C_API
#include "_surface.h"

#ifdef PG_ENABLE_ARM_NEON
    // sse2neon.h is from here: https://github.com/DLTcollab/sse2neon
    #include "include/sse2neon.h"
#else
    #if IS_SDLv1
        // MSVC uses these defines for SSE2 support for some reason
        #if defined(_M_IX86_FP) || (defined(_M_AMD64) || defined(_M_X64))
            #if (_M_IX86_FP == 2) || (defined(_M_AMD64) || defined(_M_X64))
                #define __SSE2__ 1
            #endif
        #endif
        // SDL 1 doesn't import the latest intrinsics, this should should pull
        // them all in for us
        #ifdef __SSE2__ // don't import this file on non-SSE platforms.
            #include <immintrin.h>
        #endif /* __SSE2__ */
    #endif /* IS_SDLv1 */
#endif /* PG_ENABLE_ARM_NEON */

/* The structure passed to the low level blit functions */
typedef struct
{
    int              width;
    int              height;
    Uint8           *s_pixels;
    int              s_pxskip;
    int              s_skip;
    Uint8           *d_pixels;
    int              d_pxskip;
    int              d_skip;
    SDL_PixelFormat *src;
    SDL_PixelFormat *dst;
#if IS_SDLv1
    Uint32           src_flags;
    Uint32           dst_flags;
#else /* IS_SDLv2 */
    Uint8            src_blanket_alpha;
    int              src_has_colorkey;
    Uint32           src_colorkey;
    SDL_BlendMode    src_blend;
    SDL_BlendMode    dst_blend;
#endif /* IS_SDLv2 */
} SDL_BlitInfo;

static void alphablit_alpha (SDL_BlitInfo * info);
static void alphablit_colorkey (SDL_BlitInfo * info);
static void alphablit_solid (SDL_BlitInfo * info);
static void blit_blend_add (SDL_BlitInfo * info);
static void blit_blend_sub (SDL_BlitInfo * info);
static void blit_blend_mul (SDL_BlitInfo * info);
static void blit_blend_min (SDL_BlitInfo * info);
static void blit_blend_max (SDL_BlitInfo * info);

static void blit_blend_rgba_add (SDL_BlitInfo * info);
static void blit_blend_rgba_sub (SDL_BlitInfo * info);
static void blit_blend_rgba_mul (SDL_BlitInfo * info);
static void blit_blend_rgba_min (SDL_BlitInfo * info);
static void blit_blend_rgba_max (SDL_BlitInfo * info);

static void blit_blend_premultiplied (SDL_BlitInfo * info);
#ifdef __MMX__
static void blit_blend_premultiplied_mmx (SDL_BlitInfo * info);
#endif /*  __MMX__ */
#if  defined(__MMX__) || defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)
static void blit_blend_premultiplied_sse2 (SDL_BlitInfo * info);
#endif /*defined(__MMX__) || defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)*/


static int
SoftBlitPyGame (SDL_Surface * src, SDL_Rect * srcrect,
                SDL_Surface * dst, SDL_Rect * dstrect, int the_args);
extern int  SDL_RLESurface (SDL_Surface * surface);
extern void SDL_UnRLESurface (SDL_Surface * surface, int recode);



static int
SoftBlitPyGame (SDL_Surface * src, SDL_Rect * srcrect, SDL_Surface * dst,
                SDL_Rect * dstrect, int the_args)
{
    int okay;
    int src_locked;
    int dst_locked;

    /* Everything is okay at the beginning...  */
    okay = 1;

    /* Lock the destination if it's in hardware */
    dst_locked = 0;
    if (SDL_MUSTLOCK (dst))
    {
        if (SDL_LockSurface (dst) < 0)
            okay = 0;
        else
            dst_locked = 1;
    }
    /* Lock the source if it's in hardware */
    src_locked = 0;
    if (SDL_MUSTLOCK (src))
    {
        if (SDL_LockSurface (src) < 0)
            okay = 0;
        else
            src_locked = 1;
    }

    /* Set up source and destination buffer pointers, and BLIT! */
    if (okay && srcrect->w && srcrect->h)
    {
        SDL_BlitInfo    info;

        /* Set up the blit information */
        info.width = srcrect->w;
        info.height = srcrect->h;
#if IS_SDLv1
        info.s_pixels = (Uint8 *) src->pixels + src->offset +
            (Uint16) srcrect->y * src->pitch +
            (Uint16) srcrect->x * src->format->BytesPerPixel;
        info.s_pxskip = src->format->BytesPerPixel;
        info.s_skip = src->pitch - info.width * src->format->BytesPerPixel;
        info.d_pixels = (Uint8 *) dst->pixels + dst->offset +
            (Uint16) dstrect->y * dst->pitch +
            (Uint16) dstrect->x * dst->format->BytesPerPixel;
#else /* IS_SDLv2 */
        info.s_pixels = (Uint8 *) src->pixels +
            (Uint16) srcrect->y * src->pitch +
            (Uint16) srcrect->x * src->format->BytesPerPixel;
        info.s_pxskip = src->format->BytesPerPixel;
        info.s_skip = src->pitch - info.width * src->format->BytesPerPixel;
        info.d_pixels = (Uint8 *) dst->pixels +
            (Uint16) dstrect->y * dst->pitch +
            (Uint16) dstrect->x * dst->format->BytesPerPixel;
#endif /* IS_SDLv2 */
        info.d_pxskip = dst->format->BytesPerPixel;
        info.d_skip = dst->pitch - info.width * dst->format->BytesPerPixel;
        info.src = src->format;
        info.dst = dst->format;
#if IS_SDLv1
        info.src_flags = src->flags;
        info.dst_flags = dst->flags;
#else /* IS_SDLv2 */
        SDL_GetSurfaceAlphaMod (src, &info.src_blanket_alpha);
        info.src_has_colorkey = SDL_GetColorKey (src, &info.src_colorkey) == 0;
        if (SDL_GetSurfaceBlendMode(src, &info.src_blend) ||
            SDL_GetSurfaceBlendMode(dst, &info.dst_blend)) {
            okay = 0;
        }
#endif /* IS_SDLv2 */
        if (okay){
            if (info.d_pixels > info.s_pixels)
            {
                int span = info.width * info.src->BytesPerPixel;
                Uint8 *srcpixend =
                    info.s_pixels + (info.height - 1) * src->pitch + span;

                if (info.d_pixels < srcpixend)
                {
                    int dstoffset = (info.d_pixels - info.s_pixels) % src->pitch;

                    if (dstoffset < span || dstoffset > src->pitch - span)
                    {
                        /* Overlapping Self blit with positive destination offset.
                           Reverse direction of the blit.
                        */
                        info.s_pixels = srcpixend - info.s_pxskip;
                        info.s_pxskip = -info.s_pxskip;
                        info.s_skip = -info.s_skip;
                        info.d_pixels = (info.d_pixels +
                                         (info.height - 1) * dst->pitch +
                                         span - info.d_pxskip);
                        info.d_pxskip = -info.d_pxskip;
                        info.d_skip = -info.d_skip;
                    }
                }
            }

            switch (the_args)
            {
            case 0:
            {
#if IS_SDLv1
                if (src->flags & SDL_SRCALPHA && src->format->Amask)
                    alphablit_alpha (&info);
                else if (src->flags & SDL_SRCCOLORKEY)
                    alphablit_colorkey (&info);
                else
                    alphablit_solid (&info);
                break;
#else /* IS_SDLv2 */
                if (info.src_blend != SDL_BLENDMODE_NONE &&
                    src->format->Amask) {
                    alphablit_alpha (&info);
                } else if (info.src_has_colorkey) {
                    alphablit_colorkey (&info);
                } else {
                    alphablit_solid (&info);
                }
                break;
#endif /* IS_SDLv2 */
            }
            case PYGAME_BLEND_ADD:
            {
                blit_blend_add (&info);
                break;
            }
            case PYGAME_BLEND_SUB:
            {
                blit_blend_sub (&info);
                break;
            }
            case PYGAME_BLEND_MULT:
            {
                blit_blend_mul (&info);
                break;
            }
            case PYGAME_BLEND_MIN:
            {
                blit_blend_min (&info);
                break;
            }
            case PYGAME_BLEND_MAX:
            {
                blit_blend_max (&info);
                break;
            }

            case PYGAME_BLEND_RGBA_ADD:
            {
            blit_blend_rgba_add (&info);
            break;
            }
            case PYGAME_BLEND_RGBA_SUB:
            {
                blit_blend_rgba_sub (&info);
                break;
            }
            case PYGAME_BLEND_RGBA_MULT:
            {
                blit_blend_rgba_mul (&info);
                break;
            }
            case PYGAME_BLEND_RGBA_MIN:
            {
                blit_blend_rgba_min (&info);
                break;
            }
            case PYGAME_BLEND_RGBA_MAX:
            {
                blit_blend_rgba_max (&info);
                break;
            }
            case PYGAME_BLEND_PREMULTIPLIED:
            {
        #if IS_SDLv1
                if (src->format->BytesPerPixel == 4 &&
                    dst->format->BytesPerPixel == 4 &&
                    src->format->Rmask == dst->format->Rmask &&
                    src->format->Gmask == dst->format->Gmask &&
                    src->format->Bmask == dst->format->Bmask &&
                    info.src_flags & SDL_SRCALPHA)
        #else /* IS_SDLv2 */
                if (src->format->BytesPerPixel == 4 &&
                    dst->format->BytesPerPixel == 4 &&
                    src->format->Rmask == dst->format->Rmask &&
                    src->format->Gmask == dst->format->Gmask &&
                    src->format->Bmask == dst->format->Bmask &&
                    info.src_blend != SDL_BLENDMODE_NONE)
        #endif /* IS_SDLv2 */
                {
    #if  defined(__MMX__) || defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)
        #if PG_ENABLE_ARM_NEON
                    if (SDL_HasNEON() == SDL_TRUE){
                        blit_blend_premultiplied_sse2 (&info);
                        break;
                    }
        #endif /* PG_ENABLE_ARM_NEON */
        #ifdef __SSE2__
                    if (SDL_HasSSE2()){
                        blit_blend_premultiplied_sse2 (&info);
                        break;
                    }
        #endif /* __SSE2__*/
        #ifdef __MMX__
                    if (SDL_HasMMX() == SDL_TRUE) {
                        blit_blend_premultiplied_mmx (&info);
                        break;
                    }
        #endif /*__MMX__*/
    #endif /*__MMX__ || __SSE2__ || PG_ENABLE_ARM_NEON*/
                }

                blit_blend_premultiplied (&info);
                break;
            }
            default:
            {
                SDL_SetError ("Invalid argument passed to blit.");
                okay = 0;
                break;
            }
            }
        }
    }

    /* We need to unlock the surfaces if they're locked */
    if (dst_locked)
        SDL_UnlockSurface (dst);
    if (src_locked)
        SDL_UnlockSurface (src);
    /* Blit is done! */
    return (okay ? 0 : -1);
}








/* --------------------------------------------------------- */


static void
blit_blend_rgba_add (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
    Uint32          tmp;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#else /* IS_SDLv2 */
    int             srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;
    int             dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;
#endif /* IS_SDLv2 */

    if (!dstppa)
    {
        blit_blend_add (info);
        return;
    }

#if IS_SDLv1
    if (srcbpp == 4 && dstbpp == 4 &&
        srcfmt->Rmask == dstfmt->Rmask &&
        srcfmt->Gmask == dstfmt->Gmask &&
        srcfmt->Bmask == dstfmt->Bmask &&
        srcfmt->Amask == dstfmt->Amask &&
        info->src_flags & SDL_SRCALPHA)
#else /* IS_SDLv2 */
    if (srcbpp == 4 && dstbpp == 4 &&
        srcfmt->Rmask == dstfmt->Rmask &&
        srcfmt->Gmask == dstfmt->Gmask &&
        srcfmt->Bmask == dstfmt->Bmask &&
        srcfmt->Amask == dstfmt->Amask &&
        info->src_blend != SDL_BLENDMODE_NONE)
#endif /* IS_SDLv2 */
    {
        int incr = srcpxskip > 0 ? 1 : -1;
        if (incr < 0)
        {
            src += 3;
            dst += 3;
        }
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                REPEAT_4(
                {
                    tmp = (*dst) + (*src);
                    (*dst) = (tmp <= 255 ? tmp : 255);
                    src += incr;
                    dst += incr;
                });
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_RGBA_ADD (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_RGBA_ADD (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_RGBA_ADD (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_RGBA_ADD (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void
blit_blend_rgba_sub (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
    Sint32          tmp2;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#else /* IS_SDLv2 */
    int             srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;
    int             dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;
#endif /* IS_SDLv2 */

    if (!dstppa)
    {
        blit_blend_sub (info);
        return;
    }

#if IS_SDLv1
    if (srcbpp == 4 && dstbpp == 4 &&
        srcfmt->Rmask == dstfmt->Rmask &&
        srcfmt->Gmask == dstfmt->Gmask &&
        srcfmt->Bmask == dstfmt->Bmask &&
        srcfmt->Amask == dstfmt->Amask &&
        info->src_flags & SDL_SRCALPHA)
#else /* IS_SDLv2 */
    if (srcbpp == 4 && dstbpp == 4 &&
        srcfmt->Rmask == dstfmt->Rmask &&
        srcfmt->Gmask == dstfmt->Gmask &&
        srcfmt->Bmask == dstfmt->Bmask &&
        srcfmt->Amask == dstfmt->Amask &&
        info->src_blend != SDL_BLENDMODE_NONE)
#endif /* IS_SDLv2 */
    {
        int incr = srcpxskip > 0 ? 1 : -1;
        if (incr < 0)
        {
            src += 3;
            dst += 3;
        }
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                REPEAT_4(
                {
                    tmp2 = (*dst) - (*src);
                    (*dst) = (tmp2 >= 0 ? tmp2 : 0);
                    src += incr;
                    dst += incr;
                });
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_RGBA_SUB (tmp2, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_RGBA_SUB (tmp2, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_RGBA_SUB (tmp2, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_RGBA_SUB (tmp2, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void
blit_blend_rgba_mul (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
    Uint32          tmp;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#else /* IS_SDLv2 */
    int             srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;
    int             dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;
#endif /* IS_SDLv2 */

    if (!dstppa)
    {
        blit_blend_mul (info);
        return;
    }

    if (srcbpp == 4 && dstbpp == 4 &&
        srcfmt->Rmask == dstfmt->Rmask &&
        srcfmt->Gmask == dstfmt->Gmask &&
        srcfmt->Bmask == dstfmt->Bmask &&
        srcfmt->Amask == dstfmt->Amask &&
#if IS_SDLv1
        info->src_flags & SDL_SRCALPHA)
#else /* IS_SDLv2 */
        info->src_blend != SDL_BLENDMODE_NONE)
#endif /* IS_SDLv2 */
    {
        int incr = srcpxskip > 0 ? 1 : -1;
        if (incr < 0)
        {
            src += 3;
            dst += 3;
        }
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                REPEAT_4(
                {
                    tmp = ((*dst) && (*src)) ? ((*dst) * (*src)) >> 8 : 0;
                    (*dst) = (tmp <= 255 ? tmp : 255);
                    src += incr;
                    dst += incr;
                });
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_RGBA_MULT (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_RGBA_MULT (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_RGBA_MULT (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_RGBA_MULT (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void
blit_blend_rgba_min (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#else /* IS_SDLv2 */
    int             srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;
    int             dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;
#endif /* IS_SDLv2 */

    if (!dstppa)
    {
    blit_blend_min (info);
    return;
    }

    if (srcbpp == 4 && dstbpp == 4 &&
        srcfmt->Rmask == dstfmt->Rmask &&
        srcfmt->Gmask == dstfmt->Gmask &&
        srcfmt->Bmask == dstfmt->Bmask &&
        srcfmt->Amask == dstfmt->Amask &&
#if IS_SDLv1
        info->src_flags & SDL_SRCALPHA)
#else /* IS_SDLv2 */
        info->src_blend != SDL_BLENDMODE_NONE)
#endif /* IS_SDLv2 */
    {
        int incr = srcpxskip > 0 ? 1 : -1;
        if (incr < 0)
        {
            src += 3;
            dst += 3;
        }
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                REPEAT_4(
                {
                    if ((*src) < (*dst))
                    (*dst) = (*src);
                    src += incr;
                    dst += incr;
                });
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_RGBA_MIN (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_RGBA_MIN (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_RGBA_MIN (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_RGBA_MIN (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void
blit_blend_rgba_max (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#else /* IS_SDLv2 */
    int             srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;
    int             dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;
#endif /* IS_SDLv2 */

    if (!dstppa)
    {
        blit_blend_max (info);
        return;
    }

    if (srcbpp == 4 && dstbpp == 4 &&
        srcfmt->Rmask == dstfmt->Rmask &&
        srcfmt->Gmask == dstfmt->Gmask &&
        srcfmt->Bmask == dstfmt->Bmask &&
        srcfmt->Amask == dstfmt->Amask &&
#if IS_SDLv1
        info->src_flags & SDL_SRCALPHA)
#else /* IS_SDLv2 */
        info->src_blend != SDL_BLENDMODE_NONE)
#endif /* IS_SDLv2 */
    {
        int incr = srcpxskip > 0 ? 1 : -1;
        if (incr < 0)
        {
            src += 3;
            dst += 3;
        }
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                REPEAT_4(
                {
                    if ((*src) > (*dst))
                    (*dst) = (*src);
                    src += incr;
                    dst += incr;
                });
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_RGBA_MAX (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_RGBA_MAX (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_RGBA_MAX (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_RGBA_MAX (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

#if  defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)
static void
blit_blend_premultiplied_sse2(SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint32          *srcp = (Uint32 *) info->s_pixels;
    int             srcskip = info->s_skip >> 2;
    Uint32          *dstp = (Uint32 *) info->d_pixels;
    int             dstskip = info->d_skip >> 2;
    SDL_PixelFormat *srcfmt = info->src;
    Uint32          amask = srcfmt->Amask;
    Uint64          multmask;
    Uint64          ones;

    __m128i src1, dst1, sub_dst, mm_alpha, mm_zero, multmask_128, ones_128;

    mm_zero = _mm_setzero_si128();
    multmask = 0x00FF00FF00FF00FF; // 0F0F0F0F
    multmask_128 = _mm_loadl_epi64((const __m128i *) & multmask);
    ones = 0x0001000100010001;
    ones_128 = _mm_loadl_epi64((const __m128i *) & ones);

    while (height--) {
        /* *INDENT-OFF* */
        LOOP_UNROLLED4({
        Uint32 alpha = *srcp & amask;
        if (alpha == 0) {
            /* do nothing */
        } else if (alpha == amask) {
            *dstp = *srcp;
        } else {
            src1 = _mm_cvtsi32_si128(*srcp); /* src(ARGB) -> src1 (000000000000ARGB) */
            src1 = _mm_unpacklo_epi8(src1, mm_zero); /* 000000000A0R0G0B -> src1 */

            dst1 = _mm_cvtsi32_si128(*dstp); /* dst(ARGB) -> dst1 (000000000000ARGB) */
            dst1 = _mm_unpacklo_epi8(dst1, mm_zero); /* 000000000A0R0G0B -> dst1 */

            mm_alpha = _mm_cvtsi32_si128(alpha); /* alpha -> mm_alpha (000000000000A000) */
            mm_alpha = _mm_srli_si128(mm_alpha, 3); /* mm_alpha >> ashift -> mm_alpha(000000000000000A) */
            mm_alpha = _mm_unpacklo_epi16(mm_alpha, mm_alpha); /* 0000000000000A0A -> mm_alpha */
            mm_alpha = _mm_unpacklo_epi32(mm_alpha, mm_alpha); /* 000000000A0A0A0A -> mm_alpha2 */

            /* pre-multiplied alpha blend */
            sub_dst = _mm_add_epi16(dst1, ones_128);
            sub_dst = _mm_mullo_epi16(sub_dst, mm_alpha);
            sub_dst = _mm_srli_epi16(sub_dst, 8);
            dst1 = _mm_add_epi16(src1, dst1);
            dst1 = _mm_sub_epi16(dst1, sub_dst);
            dst1 = _mm_packus_epi16(dst1, mm_zero);

            *dstp = _mm_cvtsi128_si32(dst1);
        }
        ++srcp;
        ++dstp;
        }, n, width);
        /* *INDENT-ON* */
        srcp += srcskip;
        dstp += dstskip;

    }
}
#endif /*__SSE2__ || PG_ENABLE_ARM_NEON*/

#ifdef __MMX__
/* fast ARGB888->(A)RGB888 blending with pixel alpha */
static void
blit_blend_premultiplied_mmx(SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint32          *srcp = (Uint32 *) info->s_pixels;
    int             srcskip = info->s_skip >> 2;
    Uint32          *dstp = (Uint32 *) info->d_pixels;
    int             dstskip = info->d_skip >> 2;
    SDL_PixelFormat *srcfmt = info->src;
    Uint32          amask = srcfmt->Amask;
    Uint32          ashift = srcfmt->Ashift;
    Uint64          multmask2;

    __m64 src1, dst1, mm_alpha, mm_zero, mm_alpha2;

    mm_zero = _mm_setzero_si64();       /* 0 -> mm_zero */
    multmask2 = 0x00FF00FF00FF00FFULL;

    while (height--) {
        /* *INDENT-OFF* */
        LOOP_UNROLLED4({
        Uint32 alpha = *srcp & amask;
        if (alpha == 0) {
            /* do nothing */
        } else if (alpha == amask) {
            *dstp = *srcp;
        } else {
            src1 = _mm_cvtsi32_si64(*srcp); /* src(ARGB) -> src1 (0000ARGB) */
            src1 = _mm_unpacklo_pi8(src1, mm_zero); /* 0A0R0G0B -> src1 */

            dst1 = _mm_cvtsi32_si64(*dstp); /* dst(ARGB) -> dst1 (0000ARGB) */
            dst1 = _mm_unpacklo_pi8(dst1, mm_zero); /* 0A0R0G0B -> dst1 */

            mm_alpha = _mm_cvtsi32_si64(alpha); /* alpha -> mm_alpha (0000000A) */
            mm_alpha = _mm_srli_si64(mm_alpha, ashift); /* mm_alpha >> ashift -> mm_alpha(0000000A) */
            mm_alpha = _mm_unpacklo_pi16(mm_alpha, mm_alpha); /* 00000A0A -> mm_alpha */
            mm_alpha2 = _mm_unpacklo_pi32(mm_alpha, mm_alpha); /* 0A0A0A0A -> mm_alpha2 */
            mm_alpha2 = _mm_xor_si64(mm_alpha2, *(__m64 *) & multmask2);    /* 255 - mm_alpha -> mm_alpha */

            /* pre-multiplied alpha blend */
            dst1 = _mm_mullo_pi16(dst1, mm_alpha2);
            dst1 = _mm_srli_pi16(dst1, 8);
            dst1 = _mm_add_pi16(src1, dst1);
            dst1 = _mm_packs_pu16(dst1, mm_zero);

            *dstp = _mm_cvtsi64_si32(dst1); /* dst1 -> pixel */
        }
        ++srcp;
        ++dstp;
        }, n, width);
        /* *INDENT-ON* */
        srcp += srcskip;
        dstp += dstskip;
    }
    _mm_empty();
}
#endif /*__MMX__*/

static void
blit_blend_premultiplied (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#else /* IS_SDLv2 */
    int             srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;
    int             dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;
#endif /* IS_SDLv2 */

#if IS_SDLv1
    if (srcbpp >= 3 && dstbpp >= 3 && !(info->src_flags & SDL_SRCALPHA))
#else /* IS_SDLv2 */
    if (srcbpp >= 3 && dstbpp >= 3 && info->src_blend == SDL_BLENDMODE_NONE)
#endif /* IS_SDLv2 */
    {
        size_t srcoffsetR, srcoffsetG, srcoffsetB;
        size_t dstoffsetR, dstoffsetG, dstoffsetB;
        if (srcbpp == 3)
        {
            SET_OFFSETS_24 (srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        else
        {
            SET_OFFSETS_32 (srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        if (dstbpp == 3)
        {
            SET_OFFSETS_24 (dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        else
        {
            SET_OFFSETS_32 (dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                dst[dstoffsetR] = src[srcoffsetR];
                dst[dstoffsetG] = src[srcoffsetG];
                dst[dstoffsetB] = src[srcoffsetB];

                src += srcpxskip;
                dst += dstpxskip;
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    ALPHA_BLEND_PREMULTIPLIED (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    SET_PIXELVAL (dst, dstfmt, dR, dG, dB, dA);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else if (dstbpp == 3)
        {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    ALPHA_BLEND_PREMULTIPLIED (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    ALPHA_BLEND_PREMULTIPLIED (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    ALPHA_BLEND_PREMULTIPLIED (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    SET_PIXELVAL (dst, dstfmt, dR, dG, dB, dA);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else if (dstbpp == 3)
        {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    if(sA == 0){
                        dst[offsetR] = dR;
                        dst[offsetG] = dG;
                        dst[offsetB] = dB;
                    }
                    else if(sA == 255){
                        dst[offsetR] = sR;
                        dst[offsetG] = sG;
                        dst[offsetB] = sB;
                    }
                    else{
                        ALPHA_BLEND_PREMULTIPLIED (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                        dst[offsetR] = dR;
                        dst[offsetG] = dG;
                        dst[offsetB] = dB;
                    }
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    // We can save some blending time by just copying pixels
                    // with  alphas of 255 or 0
                    if(sA == 0){
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    }
                    else if(sA == 255){
                        CREATE_PIXEL(dst, sR, sG, sB, sA, dstbpp, dstfmt);
                    }
                    else{
                        ALPHA_BLEND_PREMULTIPLIED (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    }
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}













/* --------------------------------------------------------- */


static void
blit_blend_add (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
    Uint32          tmp;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#else /* IS_SDLv2 */
    int             srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;
    int             dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;
#endif /* IS_SDLv2 */

#if IS_SDLv1
    if (srcbpp >= 3 && dstbpp >= 3 && !(info->src_flags & SDL_SRCALPHA))
#else /* IS_SDLv2 */
    if (srcbpp >= 3 && dstbpp >= 3 && info->src_blend == SDL_BLENDMODE_NONE)
#endif /* IS_SDLv2 */
    {
        size_t srcoffsetR, srcoffsetG, srcoffsetB;
        size_t dstoffsetR, dstoffsetG, dstoffsetB;
        if (srcbpp == 3)
        {
            SET_OFFSETS_24 (srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        else
        {
            SET_OFFSETS_32 (srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        if (dstbpp == 3)
        {
            SET_OFFSETS_24 (dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        else
        {
            SET_OFFSETS_32 (dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                tmp = dst[dstoffsetR] + src[srcoffsetR];
                dst[dstoffsetR] = (tmp <= 255 ? tmp : 255);
                tmp = dst[dstoffsetG] + src[srcoffsetG];
                dst[dstoffsetG] = (tmp <= 255 ? tmp : 255);
                tmp = dst[dstoffsetB] + src[srcoffsetB];
                dst[dstoffsetB] = (tmp <= 255 ? tmp : 255);
                src += srcpxskip;
                dst += dstpxskip;
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_ADD (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    SET_PIXELVAL (dst, dstfmt, dR, dG, dB, dA);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else if (dstbpp == 3)
        {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_ADD (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_ADD (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_ADD (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    SET_PIXELVAL(dst, dstfmt, dR, dG, dB, dA);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else if (dstbpp == 3)
        {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_ADD (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_ADD (tmp, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void
blit_blend_sub (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
    Sint32          tmp2;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#else /* IS_SDLv2 */
    int             srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;
    int             dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;
#endif /* IS_SDLv2 */

#if IS_SDLv1
    if (srcbpp >= 3 && dstbpp >= 3 && !(info->src_flags & SDL_SRCALPHA))
#else /* IS_SDLv2 */
    if (srcbpp >= 3 && dstbpp >= 3 && info->src_blend == SDL_BLENDMODE_NONE)
#endif /* IS_SDLv2 */
    {
        size_t srcoffsetR, srcoffsetG, srcoffsetB;
        size_t dstoffsetR, dstoffsetG, dstoffsetB;
        if (srcbpp == 3)
        {
            SET_OFFSETS_24 (srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        else
        {
            SET_OFFSETS_32 (srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        if (dstbpp == 3)
        {
            SET_OFFSETS_24 (dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        else
        {
            SET_OFFSETS_32 (dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                tmp2 = dst[dstoffsetR] - src[srcoffsetR];
                dst[dstoffsetR] = (tmp2 >= 0 ? tmp2 : 0);
                tmp2 = dst[dstoffsetG] - src[srcoffsetG];
                dst[dstoffsetG] = (tmp2 >= 0 ? tmp2 : 0);
                tmp2 = dst[dstoffsetB] - src[srcoffsetB];
                dst[dstoffsetB] = (tmp2 >= 0 ? tmp2 : 0);
                src += srcpxskip;
                dst += dstpxskip;
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_SUB (tmp2, sR, sG, sB, sA, dR, dG, dB, dA);
                    SET_PIXELVAL(dst, dstfmt, dR, dG, dB, dA);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else if (dstbpp == 3)
        {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstfmt);
                    BLEND_SUB (tmp2, sR, sG, sB, sA, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_SUB (tmp2, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_SUB (tmp2, sR, sG, sB, sA, dR, dG, dB, dA);
                    SET_PIXELVAL(dst, dstfmt, dR, dG, dB, dA);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else if (dstbpp == 3)
        {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_SUB (tmp2, sR, sG, sB, sA, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_SUB (tmp2, sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void
blit_blend_mul (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
    Uint32          tmp;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#else /* IS_SDLv2 */
    int             srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;
    int             dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;
#endif /* IS_SDLv2 */

#if IS_SDLv1
    if (srcbpp >= 3 && dstbpp >= 3 && !(info->src_flags & SDL_SRCALPHA))
#else /* IS_SDLv2 */
    if (srcbpp >= 3 && dstbpp >= 3 && info->src_blend == SDL_BLENDMODE_NONE)
#endif /* IS_SDLv2 */
    {
        size_t srcoffsetR, srcoffsetG, srcoffsetB;
        size_t dstoffsetR, dstoffsetG, dstoffsetB;
        if (srcbpp == 3)
        {
            SET_OFFSETS_24 (srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        else
        {
            SET_OFFSETS_32 (srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        if (dstbpp == 3)
        {
            SET_OFFSETS_24 (dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        else
        {
            SET_OFFSETS_32 (dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                tmp = ((dst[dstoffsetR] && src[srcoffsetR]) ?
                       (dst[dstoffsetR] * src[srcoffsetR]) >> 8 : 0);
                dst[dstoffsetR] = (tmp <= 255 ? tmp : 255);
                tmp = ((dst[dstoffsetG] && src[srcoffsetG]) ?
                       (dst[dstoffsetG] * src[srcoffsetG]) >> 8 : 0);
                dst[dstoffsetG] = (tmp <= 255 ? tmp : 255);
                tmp = ((dst[dstoffsetB] && src[srcoffsetB]) ?
                       (dst[dstoffsetB] * src[srcoffsetB]) >> 8 : 0);
                dst[dstoffsetB] = (tmp <= 255 ? tmp : 255);
                src += srcpxskip;
                dst += dstpxskip;
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_MULT (sR, sG, sB, sA, dR, dG, dB, dA);
                    SET_PIXELVAL(dst, dstfmt, dR, dG, dB, dA);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else if (dstbpp == 3)
        {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_MULT (sR, sG, sB, sA, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_MULT (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_MULT (sR, sG, sB, sA, dR, dG, dB, dA);
                    SET_PIXELVAL(dst, dstfmt, dR, dG, dB, dA);
                    *dst = (Uint8) SDL_MapRGB (dstfmt, dR, dG, dB);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else if (dstbpp == 3)
        {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_MULT (sR, sG, sB, sA, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_MULT (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void
blit_blend_min (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#else /* IS_SDLv2 */
    int             srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;
    int             dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;
#endif /* IS_SDLv2 */

#if IS_SDLv1
    if (srcbpp >= 3 && dstbpp >= 3 && !(info->src_flags & SDL_SRCALPHA))
#else /* IS_SDLv2 */
    if (srcbpp >= 3 && dstbpp >= 3 && info->src_blend == SDL_BLENDMODE_NONE)
#endif /* IS_SDLv2 */
    {
        size_t srcoffsetR, srcoffsetG, srcoffsetB;
        size_t dstoffsetR, dstoffsetG, dstoffsetB;
        if (srcbpp == 3)
        {
            SET_OFFSETS_24 (srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        else
        {
            SET_OFFSETS_32 (srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        if (dstbpp == 3)
        {
            SET_OFFSETS_24 (dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        else
        {
            SET_OFFSETS_32 (dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                if (src[srcoffsetR] < dst[dstoffsetR])
                {
                    dst[dstoffsetR] = src[srcoffsetR];
                }
                if (src[srcoffsetG] < dst[dstoffsetG])
                {
                    dst[dstoffsetG] = src[srcoffsetG];
                }
                if (src[srcoffsetB] < dst[dstoffsetB])
                {
                    dst[dstoffsetB] = src[srcoffsetB];
                }
                src += srcpxskip;
                dst += dstpxskip;
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_MIN (sR, sG, sB, sA, dR, dG, dB, dA);
                    SET_PIXELVAL(dst, dstfmt, dR, dG, dB, dA);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else if (dstbpp == 3)
        {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_MIN (sR, sG, sB, sA, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_MIN (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_MIN (sR, sG, sB, sA, dR, dG, dB, dA);
                    *dst = (Uint8) SDL_MapRGB (dstfmt, dR, dG, dB);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else if (dstbpp == 3)
        {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_MIN (sR, sG, sB, sA, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_MIN (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void
blit_blend_max (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#else /* IS_SDLv2 */
    int             srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;
    int             dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;
#endif /* IS_SDLv2 */

#if IS_SDLv1
    if (srcbpp >= 3 && dstbpp >= 3 && !(info->src_flags & SDL_SRCALPHA))
#else /* IS_SDLv2 */
    if (srcbpp >= 3 && dstbpp >= 3 && info->src_blend == SDL_BLENDMODE_NONE)
#endif /* IS_SDLv2 */
    {
        size_t srcoffsetR, srcoffsetG, srcoffsetB;
        size_t dstoffsetR, dstoffsetG, dstoffsetB;
        if (srcbpp == 3)
        {
            SET_OFFSETS_24 (srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        else
        {
            SET_OFFSETS_32 (srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        if (dstbpp == 3)
        {
            SET_OFFSETS_24 (dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        else
        {
            SET_OFFSETS_32 (dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        while (height--)
        {
            LOOP_UNROLLED4(
            {
                if (src[srcoffsetR] > dst[dstoffsetR])
                {
                    dst[dstoffsetR] = src[srcoffsetR];
                }
                if (src[srcoffsetG] > dst[dstoffsetG])
                {
                    dst[dstoffsetG] = src[srcoffsetG];
                }
                if (src[srcoffsetB] > dst[dstoffsetB])
                {
                    dst[dstoffsetB] = src[srcoffsetB];
                }
                src += srcpxskip;
                dst += dstpxskip;
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
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_MAX (sR, sG, sB, sA, dR, dG, dB, dA);
                    SET_PIXELVAL(dst, dstfmt, dR, dG, dB, dA);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else if (dstbpp == 3)
        {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_MAX (sR, sG, sB, sA, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_MAX (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
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
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    BLEND_MAX (sR, sG, sB, sA, dR, dG, dB, dA);
                    SET_PIXELVAL(dst, dstfmt, dR, dG, dB, dA);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else if (dstbpp == 3)
        {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_MAX (sR, sG, sB, sA, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    BLEND_MAX (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}





/* --------------------------------------------------------- */




























static void
alphablit_alpha (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
#if IS_SDLv1
    int             dR, dG, dB, dA, sR, sG, sB, sA;
#else /* IS_SDLv2 */
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    int             dRi, dGi, dBi, dAi, sRi, sGi, sBi, sAi;
#endif /* IS_SDLv2 */
    Uint32          pixel;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#endif /* IS_SDLv1 */

    /*
       printf ("Alpha blit with %d and %d\n", srcbpp, dstbpp);
       */

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sRi, sGi, sBi, sAi, src, srcfmt);
                    GET_PIXELVALS_1(dRi, dGi, dBi, dAi, dst, dstfmt);
                    ALPHA_BLEND (sRi, sGi, sBi, sAi, dRi, dGi, dBi, dAi);
                    CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sRi, sGi, sBi, sAi, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    SDL_GetRGBA (pixel, dstfmt, &dR, &dG, &dB, &dA);
                    dRi = dR;
                    dGi = dG;
                    dBi = dB;
                    dAi = dA;
                    ALPHA_BLEND (sRi, sGi, sBi, sAi, dRi, dGi, dBi, dAi);
                    CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
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
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    SDL_GetRGBA (pixel, srcfmt, &sR, &sG, &sB, &sA);
                    GET_PIXELVALS_1(dRi, dGi, dBi, dAi, dst, dstfmt);
                    ALPHA_BLEND (sR, sG, sB, sA, dRi, dGi, dBi, dAi);
                    CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    SDL_GetRGBA (pixel, srcfmt, &sR, &sG, &sB, &sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    SDL_GetRGBA (pixel, dstfmt, &dR, &dG, &dB, &dA);
                    dRi = dR;
                    dGi = dG;
                    dBi = dB;
                    dAi = dA;
                    ALPHA_BLEND (sR, sG, sB, sA, dRi, dGi, dBi, dAi);
                    CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void
alphablit_colorkey (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
#if IS_SDLv1
    int             dR, dG, dB, dA, sR, sG, sB, sA;
    int             alpha = srcfmt->alpha;
    Uint32          colorkey = srcfmt->colorkey;
#else /* IS_SDLv2 */
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    int             dRi, dGi, dBi, dAi, sRi, sGi, sBi, sAi;
    int             alpha = info->src_blanket_alpha;
    Uint32          colorkey = info->src_colorkey;
#endif /* IS_SDLv2 */
    Uint32          pixel;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#endif /* IS_SDLv1 */

    /*
       printf ("Colorkey blit with %d and %d\n", srcbpp, dstbpp);
       */

#if IS_SDLv2
    assert (info->src_has_colorkey);
#endif /* IS_SDLv2 */
    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    sA = (*src == colorkey) ? 0 : alpha;
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    *dst = (Uint8) SDL_MapRGB (dstfmt, dR, dG, dB);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sRi, sGi, sBi, sAi, src, srcfmt);
                    sAi = (*src == colorkey) ? 0 : alpha;
                    GET_PIXELVALS_1(dRi, dGi, dBi, dAi, dst, dstfmt);
                    ALPHA_BLEND (sRi, sGi, sBi, sAi, dRi, dGi, dBi, dAi);
                    *dst = (Uint8) SDL_MapRGBA (dstfmt, dRi, dGi, dBi, dAi);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    sA = (*src == colorkey) ? 0 : alpha;
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sRi, sGi, sBi, sAi, src, srcfmt);
                    sAi = (*src == colorkey) ? 0 : alpha;
                    GET_PIXEL (pixel, dstbpp, dst);
                    SDL_GetRGBA (pixel, dstfmt, &dR, &dG, &dB, &dA);
                    dRi = dR;
                    dGi = dG;
                    dBi = dB;
                    dAi = dA;
                    ALPHA_BLEND (sRi, sGi, sBi, sAi, dRi, dGi, dBi, dAi);
                    CREATE_PIXEL (dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
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
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    sA = (pixel == colorkey) ? 0 : alpha;
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    *dst = (Uint8) SDL_MapRGB (dstfmt, dR, dG, dB);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    SDL_GetRGBA (pixel, srcfmt, &sR, &sG, &sB, &sA);
                    sA = (pixel == colorkey) ? 0 : alpha;
                    GET_PIXELVALS_1(dRi, dGi, dBi, dAi, dst, dstfmt);
                    ALPHA_BLEND (sR, sG, sB, sA, dRi, dGi, dBi, dAi);
                    *dst = (Uint8) SDL_MapRGBA (dstfmt, dRi, dGi, dBi, dAi);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
                src += srcskip;
                dst += dstskip;
            }

        }
        else if (dstbpp == 3)
        {
            /* This is interim code until SDL can properly handle self
               blits of surfaces with blanket alpha.
               */
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    sA = (pixel == colorkey) ? 0 : alpha;
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    SDL_GetRGBA (pixel, srcfmt, &sR, &sG, &sB, &sA);
                    sA = (pixel == colorkey) ? 0 : alpha;
                    GET_PIXEL (pixel, dstbpp, dst);
                    SDL_GetRGBA (pixel, dstfmt, &dR, &dG, &dB, &dA);
                    dRi = dR;
                    dGi = dG;
                    dBi = dB;
                    dAi = dA;
                    ALPHA_BLEND (sR, sG, sB, sA, dRi, dGi, dBi, dAi);
                    dst[offsetR] = (Uint8)dRi;
                    dst[offsetG] = (Uint8)dGi;
                    dst[offsetB] = (Uint8)dBi;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--)
            {
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    sA = (pixel == colorkey) ? 0 : alpha;
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    SDL_GetRGBA (pixel, srcfmt, &sR, &sG, &sB, &sA);
                    sA = (pixel == colorkey) ? 0 : alpha;
                    GET_PIXEL (pixel, dstbpp, dst);
                    SDL_GetRGBA (pixel, dstfmt, &dR, &dG, &dB, &dA);
                    dRi = dR;
                    dGi = dG;
                    dBi = dB;
                    dAi = dA;
                    ALPHA_BLEND (sR, sG, sB, sA, dRi, dGi, dBi, dAi);
                    CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void
alphablit_solid (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;
    Uint8          *src = info->s_pixels;
    int             srcpxskip = info->s_pxskip;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstpxskip = info->d_pxskip;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
#if IS_SDLv1
    int             dR, dG, dB, dA, sR, sG, sB, sA;
    int             alpha = srcfmt->alpha;
#else /* IS_SDLv2 */
    Uint8           dR, dG, dB, dA, sR, sG, sB, sA;
    int             dRi, dGi, dBi, dAi, sRi, sGi, sBi;
    int             alpha = info->src_blanket_alpha;
#endif /* IS_SDLv2 */
    int             pixel;
#if IS_SDLv1
    int             srcppa = (info->src_flags & SDL_SRCALPHA && srcfmt->Amask);
    int             dstppa = (info->dst_flags & SDL_SRCALPHA && dstfmt->Amask);
#endif /* IS_SDLv1 */

    /*
       printf ("Solid blit with %d and %d\n", srcbpp, dstbpp);
       */

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    ALPHA_BLEND (sR, sG, sB, alpha, dR, dG, dB, dA);
                    *dst = (Uint8) SDL_MapRGB (dstfmt, dR, dG, dB);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sRi, sGi, sBi, dAi, src, srcfmt);
                    GET_PIXELVALS_1(dRi, dGi, dBi, dAi, dst, dstfmt);
                    ALPHA_BLEND (sRi, sGi, sBi, alpha, dRi, dGi, dBi, dAi);
                    *dst = (Uint8) SDL_MapRGBA (dstfmt, dRi, dGi, dBi, dAi);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sR, sG, sB, sA, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    ALPHA_BLEND (sR, sG, sB, alpha, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXELVALS_1(sRi, sGi, sBi, dAi, src, srcfmt);
                    GET_PIXEL (pixel, dstbpp, dst);
                    SDL_GetRGBA (pixel, dstfmt, &dR, &dG, &dB, &dA);
                    dRi = dR;
                    dGi = dG;
                    dBi = dB;
                    dAi = dA;
                    ALPHA_BLEND (sRi, sGi, sBi, alpha, dRi, dGi, dBi, dAi);
                    CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
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
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstfmt);
                    ALPHA_BLEND (sR, sG, sB, alpha, dR, dG, dB, dA);
                    *dst = (Uint8) SDL_MapRGB (dstfmt, dR, dG, dB);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    SDL_GetRGBA (pixel, srcfmt, &sR, &sG, &sB, &sA);
                    GET_PIXELVALS_1(dRi, dGi, dBi, dAi, dst, dstfmt);
                    ALPHA_BLEND (sR, sG, sB, alpha, dRi, dGi, dBi, dAi);
                    *dst = (Uint8) SDL_MapRGBA (dstfmt, dRi, dGi, dBi, dAi);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
                src += srcskip;
                dst += dstskip;
            }

        }
        else if (dstbpp == 3)
        {
            /* This is interim code until SDL can properly handle self
               blits of surfaces with blanket alpha.
               */
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24 (offsetR, offsetG, offsetB, dstfmt);
            while (height--)
            {
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    ALPHA_BLEND (sR, sG, sB, alpha, dR, dG, dB, dA);
                    dst[offsetR] = dR;
                    dst[offsetG] = dG;
                    dst[offsetB] = dB;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    SDL_GetRGBA (pixel, srcfmt, &sR, &sG, &sB, &sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    SDL_GetRGBA (pixel, dstfmt, &dR, &dG, &dB, &dA);
                    dRi = dR;
                    dGi = dG;
                    dBi = dB;
                    dAi = dA;
                    ALPHA_BLEND (sR, sG, sB, alpha, dRi, dGi, dBi, dAi);
                    dst[offsetR] = (Uint8)dRi;
                    dst[offsetG] = (Uint8)dGi;
                    dst[offsetB] = (Uint8)dBi;
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n, width);
#endif /* IS_SDLv2 */
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--)
            {
#if IS_SDLv1
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_PIXELVALS (dR, dG, dB, dA, pixel, dstfmt, dstppa);
                    ALPHA_BLEND (sR, sG, sB, alpha, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n ,width);
#else /* IS_SDLv2 */
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    SDL_GetRGBA (pixel, srcfmt, &sR, &sG, &sB, &sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    SDL_GetRGBA (pixel, dstfmt, &dR, &dG, &dB, &dA);
                    dRi = dR;
                    dGi = dG;
                    dBi = dB;
                    dAi = dA;
                    ALPHA_BLEND (sR, sG, sB, alpha, dRi, dGi, dBi, dAi);
                    CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                    src += srcpxskip;
                    dst += dstpxskip;
                }, n ,width);
#endif /* IS_SDLv2 */
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

/*we assume the "dst" has pixel alpha*/
int
pygame_Blit (SDL_Surface * src, SDL_Rect * srcrect,
             SDL_Surface * dst, SDL_Rect * dstrect, int the_args)
{
    SDL_Rect        fulldst;
    int             srcx, srcy, w, h;

    /* Make sure the surfaces aren't locked */
    if (!src || !dst)
    {
        SDL_SetError ("pygame_Blit: passed a NULL surface");
        return (-1);
    }
    if (src->locked || dst->locked)
    {
        SDL_SetError ("pygame_Blit: Surfaces must not be locked during blit");
        return (-1);
    }

    /* If the destination rectangle is NULL, use the entire dest surface */
    if (dstrect == NULL)
    {
        fulldst.x = fulldst.y = 0;
        dstrect = &fulldst;
    }

    /* clip the source rectangle to the source surface */
    if (srcrect)
    {
        int             maxw, maxh;

        srcx = srcrect->x;
        w = srcrect->w;
        if (srcx < 0)
        {
            w += srcx;
            dstrect->x -= srcx;
            srcx = 0;
        }
        maxw = src->w - srcx;
        if (maxw < w)
            w = maxw;

        srcy = srcrect->y;
        h = srcrect->h;
        if (srcy < 0)
        {
            h += srcy;
            dstrect->y -= srcy;
            srcy = 0;
        }
        maxh = src->h - srcy;
        if (maxh < h)
            h = maxh;

    }
    else
    {
        srcx = srcy = 0;
        w = src->w;
        h = src->h;
    }

    /* clip the destination rectangle against the clip rectangle */
    {
        SDL_Rect       *clip = &dst->clip_rect;
        int             dx, dy;

        dx = clip->x - dstrect->x;
        if (dx > 0)
        {
            w -= dx;
            dstrect->x += dx;
            srcx += dx;
        }
        dx = dstrect->x + w - clip->x - clip->w;
        if (dx > 0)
            w -= dx;

        dy = clip->y - dstrect->y;
        if (dy > 0)
        {
            h -= dy;
            dstrect->y += dy;
            srcy += dy;
        }
        dy = dstrect->y + h - clip->y - clip->h;
        if (dy > 0)
            h -= dy;
    }

    if (w > 0 && h > 0)
    {
        SDL_Rect        sr;

        sr.x = srcx;
        sr.y = srcy;
        sr.w = dstrect->w = w;
        sr.h = dstrect->h = h;
        return SoftBlitPyGame (src, &sr, dst, dstrect, the_args);
    }
    dstrect->w = dstrect->h = 0;
    return 0;
}

int
pygame_AlphaBlit (SDL_Surface * src, SDL_Rect * srcrect,
                  SDL_Surface * dst, SDL_Rect * dstrect, int the_args)
{
    return pygame_Blit (src, srcrect, dst, dstrect, the_args);
}
