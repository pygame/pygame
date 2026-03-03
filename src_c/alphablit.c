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
#include "simd_shared.h"
#include "simd_blitters.h"

static void
alphablit_alpha(SDL_BlitInfo *info);
static void
alphablit_colorkey(SDL_BlitInfo *info);
static void
alphablit_solid(SDL_BlitInfo *info);
static void
blit_blend_add(SDL_BlitInfo *info);
static void
blit_blend_sub(SDL_BlitInfo *info);
static void
blit_blend_mul(SDL_BlitInfo *info);
static void
blit_blend_min(SDL_BlitInfo *info);
static void
blit_blend_max(SDL_BlitInfo *info);

static void
blit_blend_rgba_add(SDL_BlitInfo *info);
static void
blit_blend_rgba_sub(SDL_BlitInfo *info);
static void
blit_blend_rgba_mul(SDL_BlitInfo *info);
static void
blit_blend_rgba_min(SDL_BlitInfo *info);
static void
blit_blend_rgba_max(SDL_BlitInfo *info);

static void
blit_blend_premultiplied(SDL_BlitInfo *info);

static int
SoftBlitPyGame(SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *dst,
               SDL_Rect *dstrect, int blend_flags);

static int
SoftBlitPyGame(SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *dst,
               SDL_Rect *dstrect, int blend_flags)
{
    int okay;
    int src_locked;
    int dst_locked;

    /* Everything is okay at the beginning...  */
    okay = 1;

    /* Lock the destination if it's in hardware */
    dst_locked = 0;
    if (SDL_MUSTLOCK(dst)) {
        if (!PG_LockSurface(dst)) {
            okay = 0;
        }
        else {
            dst_locked = 1;
        }
    }
    /* Lock the source if it's in hardware */
    src_locked = 0;
    if (SDL_MUSTLOCK(src)) {
        if (!PG_LockSurface(src)) {
            okay = 0;
        }
        else {
            src_locked = 1;
        }
    }

    /* Set up source and destination buffer pointers, and BLIT! */
    if (okay && srcrect->w && srcrect->h) {
        SDL_BlitInfo info;

        /* Set up the blit information */
        info.width = srcrect->w;
        info.height = srcrect->h;
        info.s_pixels = (Uint8 *)src->pixels +
                        (Uint16)srcrect->y * src->pitch +
                        (Uint16)srcrect->x * PG_SURF_BytesPerPixel(src);
        info.s_pxskip = PG_SURF_BytesPerPixel(src);
        info.s_skip = src->pitch - info.width * PG_SURF_BytesPerPixel(src);
        info.d_pixels = (Uint8 *)dst->pixels +
                        (Uint16)dstrect->y * dst->pitch +
                        (Uint16)dstrect->x * PG_SURF_BytesPerPixel(dst);
        info.d_pxskip = PG_SURF_BytesPerPixel(dst);
        info.d_skip = dst->pitch - info.width * PG_SURF_BytesPerPixel(dst);

        if (!PG_GetSurfaceDetails(src, &(info.src), &(info.src_palette))) {
            okay = 0;
        }
        if (!PG_GetSurfaceDetails(dst, &(info.dst), &(info.dst_palette))) {
            okay = 0;
        }

        SDL_GetSurfaceAlphaMod(src, &info.src_blanket_alpha);
        if ((info.src_has_colorkey = SDL_HasColorKey(src))) {
            SDL_GetColorKey(src, &info.src_colorkey);
        }
        if (!PG_GetSurfaceBlendMode(src, &info.src_blend) ||
            !PG_GetSurfaceBlendMode(dst, &info.dst_blend)) {
            okay = 0;
        }
        if (okay) {
            if (info.d_pixels > info.s_pixels) {
                int span = info.width * PG_FORMAT_BytesPerPixel(info.src);
                Uint8 *srcpixend =
                    info.s_pixels + (info.height - 1) * src->pitch + span;

                if (info.d_pixels < srcpixend) {
                    int dstoffset =
                        (info.d_pixels - info.s_pixels) % src->pitch;

                    if (dstoffset < span || dstoffset > src->pitch - span) {
                        /* Overlapping Self blit with positive destination
                           offset. Reverse direction of the blit.
                        */
                        info.s_pixels = srcpixend - info.s_pxskip;
                        info.s_pxskip = -info.s_pxskip;
                        info.s_skip = -info.s_skip;
                        info.d_pixels =
                            (info.d_pixels + (info.height - 1) * dst->pitch +
                             span - info.d_pxskip);
                        info.d_pxskip = -info.d_pxskip;
                        info.d_skip = -info.d_skip;
                    }
                }
            }
            /* Convert alpha multiply blends to regular blends if either of
             the surfaces don't have alpha channels */
            if (blend_flags == PYGAME_BLEND_RGBA_MULT &&
                (info.src_blend == SDL_BLENDMODE_NONE ||
                 info.dst_blend == SDL_BLENDMODE_NONE)) {
                blend_flags = PYGAME_BLEND_MULT;
            }

            switch (blend_flags) {
                case 0: {
                    if (info.src_blend != SDL_BLENDMODE_NONE &&
                        info.src->Amask) {
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                        if (PG_SURF_BytesPerPixel(src) == 4 &&
                            PG_SURF_BytesPerPixel(dst) == 4 &&
                            info.src->Rmask == info.dst->Rmask &&
                            info.src->Gmask == info.dst->Gmask &&
                            info.src->Bmask == info.dst->Bmask) {
                            /* If our source and destination are the same ARGB
                               32bit format we can use SSE2/NEON/AVX2 to speed
                               up the blend */
                            if (pg_has_avx2() && (src != dst)) {
                                if (info.src_blanket_alpha != 255) {
                                    alphablit_alpha_avx2_argb_surf_alpha(
                                        &info);
                                }
                                else if (SDL_ISPIXELFORMAT_ALPHA(
                                             PG_SURF_FORMATENUM(dst)) &&
                                         info.dst_blend !=
                                             SDL_BLENDMODE_NONE) {
                                    alphablit_alpha_avx2_argb_no_surf_alpha(
                                        &info);
                                }
                                else {
                                    alphablit_alpha_avx2_argb_no_surf_alpha_opaque_dst(
                                        &info);
                                }
                                break;
                            }
#if PG_ENABLE_SSE_NEON
                            if ((pg_HasSSE_NEON()) && (src != dst)) {
                                if (info.src_blanket_alpha != 255) {
                                    alphablit_alpha_sse2_argb_surf_alpha(
                                        &info);
                                }
                                else if (SDL_ISPIXELFORMAT_ALPHA(
                                             PG_SURF_FORMATENUM(dst)) &&
                                         info.dst_blend !=
                                             SDL_BLENDMODE_NONE) {
                                    alphablit_alpha_sse2_argb_no_surf_alpha(
                                        &info);
                                }
                                else {
                                    alphablit_alpha_sse2_argb_no_surf_alpha_opaque_dst(
                                        &info);
                                }
                                break;
                            }
#endif /* PG_ENABLE_SSE_NEON */
                        }
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */
                        alphablit_alpha(&info);
                    }
                    else if (info.src_has_colorkey) {
                        alphablit_colorkey(&info);
                    }
                    else {
                        alphablit_solid(&info);
                    }
                    break;
                }
                case PYGAME_BLEND_ADD: {
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        !(info.src->Amask != 0 && info.dst->Amask != 0 &&
                          info.src->Amask != info.dst->Amask) &&
                        pg_has_avx2() && (src != dst)) {
                        blit_blend_rgb_add_avx2(&info);
                        break;
                    }
#if PG_ENABLE_SSE_NEON
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        !(info.src->Amask != 0 && info.dst->Amask != 0 &&
                          info.src->Amask != info.dst->Amask) &&
                        pg_HasSSE_NEON() && (src != dst)) {
                        blit_blend_rgb_add_sse2(&info);
                        break;
                    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */
                    blit_blend_add(&info);
                    break;
                }
                case PYGAME_BLEND_SUB: {
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        !(info.src->Amask != 0 && info.dst->Amask != 0 &&
                          info.src->Amask != info.dst->Amask) &&
                        pg_has_avx2() && (src != dst)) {
                        blit_blend_rgb_sub_avx2(&info);
                        break;
                    }
#if PG_ENABLE_SSE_NEON
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        !(info.src->Amask != 0 && info.dst->Amask != 0 &&
                          info.src->Amask != info.dst->Amask) &&
                        pg_HasSSE_NEON() && (src != dst)) {
                        blit_blend_rgb_sub_sse2(&info);
                        break;
                    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */
                    blit_blend_sub(&info);
                    break;
                }
                case PYGAME_BLEND_MULT: {
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        !(info.src->Amask != 0 && info.dst->Amask != 0 &&
                          info.src->Amask != info.dst->Amask) &&
                        pg_has_avx2() && (src != dst)) {
                        blit_blend_rgb_mul_avx2(&info);
                        break;
                    }
#if PG_ENABLE_SSE_NEON
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        !(info.src->Amask != 0 && info.dst->Amask != 0 &&
                          info.src->Amask != info.dst->Amask) &&
                        pg_HasSSE_NEON() && (src != dst)) {
                        blit_blend_rgb_mul_sse2(&info);
                        break;
                    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */
                    blit_blend_mul(&info);
                    break;
                }
                case PYGAME_BLEND_MIN: {
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        !(info.src->Amask != 0 && info.dst->Amask != 0 &&
                          info.src->Amask != info.dst->Amask) &&
                        pg_has_avx2() && (src != dst)) {
                        blit_blend_rgb_min_avx2(&info);
                        break;
                    }
#if PG_ENABLE_SSE_NEON
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        !(info.src->Amask != 0 && info.dst->Amask != 0 &&
                          info.src->Amask != info.dst->Amask) &&
                        pg_HasSSE_NEON() && (src != dst)) {
                        blit_blend_rgb_min_sse2(&info);
                        break;
                    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */
                    blit_blend_min(&info);
                    break;
                }
                case PYGAME_BLEND_MAX: {
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        !(info.src->Amask != 0 && info.dst->Amask != 0 &&
                          info.src->Amask != info.dst->Amask) &&
                        pg_has_avx2() && (src != dst)) {
                        blit_blend_rgb_max_avx2(&info);
                        break;
                    }
#if PG_ENABLE_SSE_NEON
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        !(info.src->Amask != 0 && info.dst->Amask != 0 &&
                          info.src->Amask != info.dst->Amask) &&
                        pg_HasSSE_NEON() && (src != dst)) {
                        blit_blend_rgb_max_sse2(&info);
                        break;
                    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */
                    blit_blend_max(&info);
                    break;
                }

                case PYGAME_BLEND_RGBA_ADD: {
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        info.src_blend != SDL_BLENDMODE_NONE &&
                        pg_has_avx2() && (src != dst)) {
                        blit_blend_rgba_add_avx2(&info);
                        break;
                    }
#if PG_ENABLE_SSE_NEON
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        info.src_blend != SDL_BLENDMODE_NONE &&
                        pg_HasSSE_NEON() && (src != dst)) {
                        blit_blend_rgba_add_sse2(&info);
                        break;
                    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */
                    blit_blend_rgba_add(&info);
                    break;
                }
                case PYGAME_BLEND_RGBA_SUB: {
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        info.src_blend != SDL_BLENDMODE_NONE &&
                        pg_has_avx2() && (src != dst)) {
                        blit_blend_rgba_sub_avx2(&info);
                        break;
                    }
#if PG_ENABLE_SSE_NEON
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        info.src_blend != SDL_BLENDMODE_NONE &&
                        pg_HasSSE_NEON() && (src != dst)) {
                        blit_blend_rgba_sub_sse2(&info);
                        break;
                    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */
                    blit_blend_rgba_sub(&info);
                    break;
                }
                case PYGAME_BLEND_RGBA_MULT: {
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        info.src_blend != SDL_BLENDMODE_NONE &&
                        pg_has_avx2() && (src != dst)) {
                        blit_blend_rgba_mul_avx2(&info);
                        break;
                    }
#if PG_ENABLE_SSE_NEON
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        info.src_blend != SDL_BLENDMODE_NONE &&
                        pg_HasSSE_NEON() && (src != dst)) {
                        blit_blend_rgba_mul_sse2(&info);
                        break;
                    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */
                    blit_blend_rgba_mul(&info);
                    break;
                }
                case PYGAME_BLEND_RGBA_MIN: {
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        info.src_blend != SDL_BLENDMODE_NONE &&
                        pg_has_avx2() && (src != dst)) {
                        blit_blend_rgba_min_avx2(&info);
                        break;
                    }
#if PG_ENABLE_SSE_NEON
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        info.src_blend != SDL_BLENDMODE_NONE &&
                        pg_HasSSE_NEON() && (src != dst)) {
                        blit_blend_rgba_min_sse2(&info);
                        break;
                    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */
                    blit_blend_rgba_min(&info);
                    break;
                }
                case PYGAME_BLEND_RGBA_MAX: {
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        info.src_blend != SDL_BLENDMODE_NONE &&
                        pg_has_avx2() && (src != dst)) {
                        blit_blend_rgba_max_avx2(&info);
                        break;
                    }
#if PG_ENABLE_SSE_NEON
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        info.src_blend != SDL_BLENDMODE_NONE &&
                        pg_HasSSE_NEON() && (src != dst)) {
                        blit_blend_rgba_max_sse2(&info);
                        break;
                    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */
                    blit_blend_rgba_max(&info);
                    break;
                }
                case PYGAME_BLEND_PREMULTIPLIED: {
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        info.src_blend != SDL_BLENDMODE_NONE &&
                        pg_has_avx2() && (src != dst)) {
                        blit_blend_premultiplied_avx2(&info);
                        break;
                    }
#if PG_ENABLE_SSE_NEON
                    if (PG_SURF_BytesPerPixel(src) == 4 &&
                        PG_SURF_BytesPerPixel(dst) == 4 &&
                        info.src->Rmask == info.dst->Rmask &&
                        info.src->Gmask == info.dst->Gmask &&
                        info.src->Bmask == info.dst->Bmask &&
                        info.src->Amask == 0xFF000000 &&
                        info.src_blend != SDL_BLENDMODE_NONE &&
                        pg_HasSSE_NEON() && (src != dst)) {
                        blit_blend_premultiplied_sse2(&info);
                        break;
                    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */

                    blit_blend_premultiplied(&info);
                    break;
                }
                default: {
                    SDL_SetError("Invalid argument passed to blit.");
                    okay = 0;
                    break;
                }
            }
        }
    }

    /* We need to unlock the surfaces if they're locked */
    if (dst_locked) {
        SDL_UnlockSurface(dst);
    }
    if (src_locked) {
        SDL_UnlockSurface(src);
    }
    /* Blit is done! */
    return (okay ? 0 : -1);
}

/* --------------------------------------------------------- */

#define SETUP_BASE                                \
    int n;                                        \
    int width = info->width;                      \
    int height = info->height;                    \
    Uint8 *src = info->s_pixels;                  \
    int srcpxskip = info->s_pxskip;               \
    int srcskip = info->s_skip;                   \
    Uint8 *dst = info->d_pixels;                  \
    int dstpxskip = info->d_pxskip;               \
    int dstskip = info->d_skip;                   \
    PG_PixelFormat *srcfmt = info->src;           \
    SDL_Palette *srcpal = info->src_palette;      \
    PG_PixelFormat *dstfmt = info->dst;           \
    SDL_Palette *dstpal = info->dst_palette;      \
    int srcbpp = PG_FORMAT_BytesPerPixel(srcfmt); \
    int dstbpp = PG_FORMAT_BytesPerPixel(dstfmt); \
    Uint8 dR, dG, dB, dA, sR, sG, sB, sA;         \
    Uint32 pixel;

#define SETUP_BLIT_BLEND                                                 \
    SETUP_BASE                                                           \
    int srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask; \
    int dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;

/* --------------------------------------------------------- */

#define BLIT_BLEND_RGBA_OP(operation, BLEND_CODE,                            \
                           BLEND_CODE_32BIT_SPECIALIZED)                     \
    SETUP_BLIT_BLEND                                                         \
    if (!dstppa) {                                                           \
        blit_blend_##operation(info);                                        \
        return;                                                              \
    }                                                                        \
    if (srcbpp == 4 && dstbpp == 4 && srcfmt->Rmask == dstfmt->Rmask &&      \
        srcfmt->Gmask == dstfmt->Gmask && srcfmt->Bmask == dstfmt->Bmask &&  \
        srcfmt->Amask == dstfmt->Amask &&                                    \
        info->src_blend != SDL_BLENDMODE_NONE) {                             \
        int incr = srcpxskip > 0 ? 1 : -1;                                   \
        if (incr < 0) {                                                      \
            src += 3;                                                        \
            dst += 3;                                                        \
        }                                                                    \
        while (height--) {                                                   \
            LOOP_UNROLLED4(                                                  \
                {                                                            \
                    REPEAT_4({                                               \
                        BLEND_CODE_32BIT_SPECIALIZED                         \
                        src += incr;                                         \
                        dst += incr;                                         \
                    });                                                      \
                },                                                           \
                n, width);                                                   \
            src += srcskip;                                                  \
            dst += dstskip;                                                  \
        }                                                                    \
        return;                                                              \
    }                                                                        \
    if (srcbpp == 1) {                                                       \
        if (dstbpp == 1) {                                                   \
            while (height--) {                                               \
                LOOP_UNROLLED4(                                              \
                    {                                                        \
                        GET_PIXELVALS_1(sR, sG, sB, sA, src, srcpal);        \
                        GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstpal);        \
                        BLEND_CODE                                           \
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);   \
                        src += srcpxskip;                                    \
                        dst += dstpxskip;                                    \
                    },                                                       \
                    n, width);                                               \
                src += srcskip;                                              \
                dst += dstskip;                                              \
            }                                                                \
        }                                                                    \
        else /* dstbpp > 1 */                                                \
        {                                                                    \
            while (height--) {                                               \
                LOOP_UNROLLED4(                                              \
                    {                                                        \
                        GET_PIXELVALS_1(sR, sG, sB, sA, src, srcpal);        \
                        GET_PIXEL(pixel, dstbpp, dst);                       \
                        GET_PIXELVALS(dR, dG, dB, dA, pixel, dstfmt, dstpal, \
                                      dstppa);                               \
                        BLEND_CODE                                           \
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);   \
                        src += srcpxskip;                                    \
                        dst += dstpxskip;                                    \
                    },                                                       \
                    n, width);                                               \
                src += srcskip;                                              \
                dst += dstskip;                                              \
            }                                                                \
        }                                                                    \
    }                                                                        \
    else /* srcbpp > 1 */                                                    \
    {                                                                        \
        if (dstbpp == 1) {                                                   \
            while (height--) {                                               \
                LOOP_UNROLLED4(                                              \
                    {                                                        \
                        GET_PIXEL(pixel, srcbpp, src);                       \
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, srcfmt, srcpal, \
                                      srcppa);                               \
                        GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstpal);        \
                        BLEND_CODE                                           \
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);   \
                        src += srcpxskip;                                    \
                        dst += dstpxskip;                                    \
                    },                                                       \
                    n, width);                                               \
                src += srcskip;                                              \
                dst += dstskip;                                              \
            }                                                                \
        }                                                                    \
        else /* dstbpp > 1 */                                                \
        {                                                                    \
            while (height--) {                                               \
                LOOP_UNROLLED4(                                              \
                    {                                                        \
                        GET_PIXEL(pixel, srcbpp, src);                       \
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, srcfmt, srcpal, \
                                      srcppa);                               \
                        GET_PIXEL(pixel, dstbpp, dst);                       \
                        GET_PIXELVALS(dR, dG, dB, dA, pixel, dstfmt, dstpal, \
                                      dstppa);                               \
                        BLEND_CODE                                           \
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);   \
                        src += srcpxskip;                                    \
                        dst += dstpxskip;                                    \
                    },                                                       \
                    n, width);                                               \
                src += srcskip;                                              \
                dst += dstskip;                                              \
            }                                                                \
        }                                                                    \
    }

static void
blit_blend_rgba_add(SDL_BlitInfo *info)
{
    Uint32 tmp;
    BLIT_BLEND_RGBA_OP(add,
                       BLEND_RGBA_ADD(tmp, sR, sG, sB, sA, dR, dG, dB, dA), {
                           tmp = (*dst) + (*src);
                           (*dst) = (tmp <= 255 ? tmp : 255);
                       });
}

static void
blit_blend_rgba_sub(SDL_BlitInfo *info)
{
    Sint32 tmp;
    BLIT_BLEND_RGBA_OP(sub,
                       BLEND_RGBA_SUB(tmp, sR, sG, sB, sA, dR, dG, dB, dA), {
                           tmp = (*dst) - (*src);
                           (*dst) = (tmp >= 0 ? tmp : 0);
                       });
}

static void
blit_blend_rgba_mul(SDL_BlitInfo *info)
{
    Sint32 tmp;
    BLIT_BLEND_RGBA_OP(mul, BLEND_RGBA_MULT(sR, sG, sB, sA, dR, dG, dB, dA), {
        tmp = ((*dst) && (*src)) ? (((*dst) * (*src)) + 255) >> 8 : 0;
        (*dst) = (tmp <= 255 ? tmp : 255);
    });
}

static void
blit_blend_rgba_min(SDL_BlitInfo *info)
{
    BLIT_BLEND_RGBA_OP(min, BLEND_RGBA_MIN(sR, sG, sB, sA, dR, dG, dB, dA), {
        if ((*src) < (*dst)) {
            (*dst) = (*src);
        }
    });
}

static void
blit_blend_rgba_max(SDL_BlitInfo *info)
{
    BLIT_BLEND_RGBA_OP(max, BLEND_RGBA_MAX(sR, sG, sB, sA, dR, dG, dB, dA), {
        if ((*src) > (*dst)) {
            (*dst) = (*src);
        }
    });
}

static void
blit_blend_premultiplied(SDL_BlitInfo *info)
{
    SETUP_BLIT_BLEND
    if (srcbpp >= 3 && dstbpp >= 3 && info->src_blend == SDL_BLENDMODE_NONE) {
        size_t srcoffsetR, srcoffsetG, srcoffsetB;
        size_t dstoffsetR, dstoffsetG, dstoffsetB;
        if (srcbpp == 3) {
            SET_OFFSETS_24(srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        else {
            SET_OFFSETS_32(srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);
        }
        if (dstbpp == 3) {
            SET_OFFSETS_24(dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        else {
            SET_OFFSETS_32(dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);
        }
        while (height--) {
            LOOP_UNROLLED4(
                {
                    dst[dstoffsetR] = src[srcoffsetR];
                    dst[dstoffsetG] = src[srcoffsetG];
                    dst[dstoffsetB] = src[srcoffsetB];

                    src += srcpxskip;
                    dst += dstpxskip;
                },
                n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

    if (srcbpp == 1) {
        if (dstbpp == 1) {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, src, srcpal);
                        GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstpal);
                        ALPHA_BLEND_PREMULTIPLIED(tmp, sR, sG, sB, sA, dR, dG,
                                                  dB, dA);
                        SET_PIXELVAL(dst, dstfmt, dstpal, dR, dG, dB, dA);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else if (dstbpp == 3) {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24(offsetR, offsetG, offsetB, dstfmt);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, src, srcpal);
                        GET_PIXEL(pixel, dstbpp, dst);
                        GET_PIXELVALS(dR, dG, dB, dA, pixel, dstfmt, dstpal,
                                      dstppa);
                        ALPHA_BLEND_PREMULTIPLIED(tmp, sR, sG, sB, sA, dR, dG,
                                                  dB, dA);
                        dst[offsetR] = dR;
                        dst[offsetG] = dG;
                        dst[offsetB] = dB;
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sR, sG, sB, sA, src, srcpal);
                        GET_PIXEL(pixel, dstbpp, dst);
                        GET_PIXELVALS(dR, dG, dB, dA, pixel, dstfmt, dstpal,
                                      dstppa);
                        ALPHA_BLEND_PREMULTIPLIED(tmp, sR, sG, sB, sA, dR, dG,
                                                  dB, dA);
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1) {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, srcbpp, src);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, srcfmt, srcpal,
                                      srcppa);
                        GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstpal);
                        ALPHA_BLEND_PREMULTIPLIED(tmp, sR, sG, sB, sA, dR, dG,
                                                  dB, dA);
                        SET_PIXELVAL(dst, dstfmt, dstpal, dR, dG, dB, dA);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else if (dstbpp == 3) {
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24(offsetR, offsetG, offsetB, dstfmt);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, srcbpp, src);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, srcfmt, srcpal,
                                      srcppa);
                        GET_PIXEL(pixel, dstbpp, dst);
                        GET_PIXELVALS(dR, dG, dB, dA, pixel, dstfmt, dstpal,
                                      dstppa);
                        if (sA == 0) {
                            dst[offsetR] = dR;
                            dst[offsetG] = dG;
                            dst[offsetB] = dB;
                        }
                        else if (sA == 255) {
                            dst[offsetR] = sR;
                            dst[offsetG] = sG;
                            dst[offsetB] = sB;
                        }
                        else {
                            ALPHA_BLEND_PREMULTIPLIED(tmp, sR, sG, sB, sA, dR,
                                                      dG, dB, dA);
                            dst[offsetR] = dR;
                            dst[offsetG] = dG;
                            dst[offsetB] = dB;
                        }
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, srcbpp, src);
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, srcfmt, srcpal,
                                      srcppa);
                        GET_PIXEL(pixel, dstbpp, dst);
                        GET_PIXELVALS(dR, dG, dB, dA, pixel, dstfmt, dstpal,
                                      dstppa);
                        // We can save some blending time by just copying
                        // pixels with  alphas of 255 or 0
                        if (sA == 0) {
                            CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                        }
                        else if (sA == 255) {
                            CREATE_PIXEL(dst, sR, sG, sB, sA, dstbpp, dstfmt);
                        }
                        else {
                            ALPHA_BLEND_PREMULTIPLIED(tmp, sR, sG, sB, sA, dR,
                                                      dG, dB, dA);
                            CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                        }
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

/* --------------------------------------------------------- */

#define BLIT_BLEND_OP(BLEND_CODE, BLEND_CODE_32BIT_SPECIALIZED)              \
    int n;                                                                   \
    int width = info->width;                                                 \
    int height = info->height;                                               \
    Uint8 *src = info->s_pixels;                                             \
    int srcpxskip = info->s_pxskip;                                          \
    int srcskip = info->s_skip;                                              \
    Uint8 *dst = info->d_pixels;                                             \
    int dstpxskip = info->d_pxskip;                                          \
    int dstskip = info->d_skip;                                              \
    PG_PixelFormat *srcfmt = info->src;                                      \
    SDL_Palette *srcpal = info->src_palette;                                 \
    PG_PixelFormat *dstfmt = info->dst;                                      \
    SDL_Palette *dstpal = info->dst_palette;                                 \
    int srcbpp = PG_FORMAT_BytesPerPixel(srcfmt);                            \
    int dstbpp = PG_FORMAT_BytesPerPixel(dstfmt);                            \
    Uint8 dR, dG, dB, dA, sR, sG, sB, sA;                                    \
    Uint32 pixel;                                                            \
    int srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;     \
    int dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;     \
    if (srcbpp >= 3 && dstbpp >= 3 &&                                        \
        info->src_blend == SDL_BLENDMODE_NONE) {                             \
        size_t srcoffsetR, srcoffsetG, srcoffsetB;                           \
        size_t dstoffsetR, dstoffsetG, dstoffsetB;                           \
        if (srcbpp == 3) {                                                   \
            SET_OFFSETS_24(srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);      \
        }                                                                    \
        else {                                                               \
            SET_OFFSETS_32(srcoffsetR, srcoffsetG, srcoffsetB, srcfmt);      \
        }                                                                    \
        if (dstbpp == 3) {                                                   \
            SET_OFFSETS_24(dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);      \
        }                                                                    \
        else {                                                               \
            SET_OFFSETS_32(dstoffsetR, dstoffsetG, dstoffsetB, dstfmt);      \
        }                                                                    \
        while (height--) {                                                   \
            LOOP_UNROLLED4(                                                  \
                {                                                            \
                    BLEND_CODE_32BIT_SPECIALIZED                             \
                    src += srcpxskip;                                        \
                    dst += dstpxskip;                                        \
                },                                                           \
                n, width);                                                   \
            src += srcskip;                                                  \
            dst += dstskip;                                                  \
        }                                                                    \
        return;                                                              \
    }                                                                        \
    if (srcbpp == 1) {                                                       \
        if (dstbpp == 1) {                                                   \
            while (height--) {                                               \
                LOOP_UNROLLED4(                                              \
                    {                                                        \
                        GET_PIXELVALS_1(sR, sG, sB, sA, src, srcpal);        \
                        GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstpal);        \
                        BLEND_CODE                                           \
                        SET_PIXELVAL(dst, dstfmt, dstpal, dR, dG, dB, dA);   \
                        src += srcpxskip;                                    \
                        dst += dstpxskip;                                    \
                    },                                                       \
                    n, width);                                               \
                src += srcskip;                                              \
                dst += dstskip;                                              \
            }                                                                \
        }                                                                    \
        else if (dstbpp == 3) {                                              \
            size_t offsetR, offsetG, offsetB;                                \
            SET_OFFSETS_24(offsetR, offsetG, offsetB, dstfmt);               \
            while (height--) {                                               \
                LOOP_UNROLLED4(                                              \
                    {                                                        \
                        GET_PIXELVALS_1(sR, sG, sB, sA, src, srcpal);        \
                        GET_PIXEL(pixel, dstbpp, dst);                       \
                        GET_PIXELVALS(dR, dG, dB, dA, pixel, dstfmt, dstpal, \
                                      dstppa);                               \
                        BLEND_CODE                                           \
                        dst[offsetR] = dR;                                   \
                        dst[offsetG] = dG;                                   \
                        dst[offsetB] = dB;                                   \
                        src += srcpxskip;                                    \
                        dst += dstpxskip;                                    \
                    },                                                       \
                    n, width);                                               \
                src += srcskip;                                              \
                dst += dstskip;                                              \
            }                                                                \
        }                                                                    \
        else /* even dstbpp */                                               \
        {                                                                    \
            while (height--) {                                               \
                LOOP_UNROLLED4(                                              \
                    {                                                        \
                        GET_PIXELVALS_1(sR, sG, sB, sA, src, srcpal);        \
                        GET_PIXEL(pixel, dstbpp, dst);                       \
                        GET_PIXELVALS(dR, dG, dB, dA, pixel, dstfmt, dstpal, \
                                      dstppa);                               \
                        BLEND_CODE                                           \
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);   \
                        src += srcpxskip;                                    \
                        dst += dstpxskip;                                    \
                    },                                                       \
                    n, width);                                               \
                src += srcskip;                                              \
                dst += dstskip;                                              \
            }                                                                \
        }                                                                    \
    }                                                                        \
    else /* srcbpp > 1 */                                                    \
    {                                                                        \
        if (dstbpp == 1) {                                                   \
            while (height--) {                                               \
                LOOP_UNROLLED4(                                              \
                    {                                                        \
                        GET_PIXEL(pixel, srcbpp, src);                       \
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, srcfmt, srcpal, \
                                      srcppa);                               \
                        GET_PIXELVALS_1(dR, dG, dB, dA, dst, dstpal);        \
                        BLEND_CODE                                           \
                        SET_PIXELVAL(dst, dstfmt, dstpal, dR, dG, dB, dA);   \
                        src += srcpxskip;                                    \
                        dst += dstpxskip;                                    \
                    },                                                       \
                    n, width);                                               \
                src += srcskip;                                              \
                dst += dstskip;                                              \
            }                                                                \
        }                                                                    \
        else if (dstbpp == 3) {                                              \
            size_t offsetR, offsetG, offsetB;                                \
            SET_OFFSETS_24(offsetR, offsetG, offsetB, dstfmt);               \
            while (height--) {                                               \
                LOOP_UNROLLED4(                                              \
                    {                                                        \
                        GET_PIXEL(pixel, srcbpp, src);                       \
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, srcfmt, srcpal, \
                                      srcppa);                               \
                        GET_PIXEL(pixel, dstbpp, dst);                       \
                        GET_PIXELVALS(dR, dG, dB, dA, pixel, dstfmt, dstpal, \
                                      dstppa);                               \
                        BLEND_CODE                                           \
                        dst[offsetR] = dR;                                   \
                        dst[offsetG] = dG;                                   \
                        dst[offsetB] = dB;                                   \
                        src += srcpxskip;                                    \
                        dst += dstpxskip;                                    \
                    },                                                       \
                    n, width);                                               \
                src += srcskip;                                              \
                dst += dstskip;                                              \
            }                                                                \
        }                                                                    \
        else /* even dstbpp */                                               \
        {                                                                    \
            while (height--) {                                               \
                LOOP_UNROLLED4(                                              \
                    {                                                        \
                        GET_PIXEL(pixel, srcbpp, src);                       \
                        GET_PIXELVALS(sR, sG, sB, sA, pixel, srcfmt, srcpal, \
                                      srcppa);                               \
                        GET_PIXEL(pixel, dstbpp, dst);                       \
                        GET_PIXELVALS(dR, dG, dB, dA, pixel, dstfmt, dstpal, \
                                      dstppa);                               \
                        BLEND_CODE                                           \
                        CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);   \
                        src += srcpxskip;                                    \
                        dst += dstpxskip;                                    \
                    },                                                       \
                    n, width);                                               \
                src += srcskip;                                              \
                dst += dstskip;                                              \
            }                                                                \
        }                                                                    \
    }

static void
blit_blend_add(SDL_BlitInfo *info)
{
    Uint32 tmp;
    BLIT_BLEND_OP(BLEND_ADD(tmp, sR, sG, sB, sA, dR, dG, dB, dA), {
        tmp = dst[dstoffsetR] + src[srcoffsetR];
        dst[dstoffsetR] = (tmp <= 255 ? tmp : 255);
        tmp = dst[dstoffsetG] + src[srcoffsetG];
        dst[dstoffsetG] = (tmp <= 255 ? tmp : 255);
        tmp = dst[dstoffsetB] + src[srcoffsetB];
        dst[dstoffsetB] = (tmp <= 255 ? tmp : 255);
    });
}

static void
blit_blend_sub(SDL_BlitInfo *info)
{
    Sint32 tmp;
    BLIT_BLEND_OP(BLEND_SUB(tmp, sR, sG, sB, sA, dR, dG, dB, dA), {
        tmp = dst[dstoffsetR] - src[srcoffsetR];
        dst[dstoffsetR] = (tmp >= 0 ? tmp : 0);
        tmp = dst[dstoffsetG] - src[srcoffsetG];
        dst[dstoffsetG] = (tmp >= 0 ? tmp : 0);
        tmp = dst[dstoffsetB] - src[srcoffsetB];
        dst[dstoffsetB] = (tmp >= 0 ? tmp : 0);
    });
}

static void
blit_blend_mul(SDL_BlitInfo *info)
{
    Sint32 tmp;
    BLIT_BLEND_OP(BLEND_MULT(sR, sG, sB, sA, dR, dG, dB, dA), {
        tmp = ((dst[dstoffsetR] && src[srcoffsetR])
                   ? ((dst[dstoffsetR] * src[srcoffsetR]) + 255) >> 8
                   : 0);
        dst[dstoffsetR] = (tmp <= 255 ? tmp : 255);
        tmp = ((dst[dstoffsetG] && src[srcoffsetG])
                   ? ((dst[dstoffsetG] * src[srcoffsetG]) + 255) >> 8
                   : 0);
        dst[dstoffsetG] = (tmp <= 255 ? tmp : 255);
        tmp = ((dst[dstoffsetB] && src[srcoffsetB])
                   ? ((dst[dstoffsetB] * src[srcoffsetB]) + 255) >> 8
                   : 0);
        dst[dstoffsetB] = (tmp <= 255 ? tmp : 255);
    });
}

static void
blit_blend_min(SDL_BlitInfo *info)
{
    BLIT_BLEND_OP(BLEND_MIN(sR, sG, sB, sA, dR, dG, dB, dA), {
        if (src[srcoffsetR] < dst[dstoffsetR]) {
            dst[dstoffsetR] = src[srcoffsetR];
        }
        if (src[srcoffsetG] < dst[dstoffsetG]) {
            dst[dstoffsetG] = src[srcoffsetG];
        }
        if (src[srcoffsetB] < dst[dstoffsetB]) {
            dst[dstoffsetB] = src[srcoffsetB];
        }
    });
}

static void
blit_blend_max(SDL_BlitInfo *info)
{
    BLIT_BLEND_OP(BLEND_MAX(sR, sG, sB, sA, dR, dG, dB, dA), {
        if (src[srcoffsetR] > dst[dstoffsetR]) {
            dst[dstoffsetR] = src[srcoffsetR];
        }
        if (src[srcoffsetG] > dst[dstoffsetG]) {
            dst[dstoffsetG] = src[srcoffsetG];
        }
        if (src[srcoffsetB] > dst[dstoffsetB]) {
            dst[dstoffsetB] = src[srcoffsetB];
        }
    });
}

/* --------------------------------------------------------- */

#define SETUP_ALPHABLIT                       \
    SETUP_BASE                                \
    Uint32 dRi, dGi, dBi, dAi, sRi, sGi, sBi; \
    int alpha = info->src_blanket_alpha;

static void
alphablit_alpha(SDL_BlitInfo *info)
{
    SETUP_ALPHABLIT
    Uint32 sAi;

    if (srcbpp == 1) {
        if (dstbpp == 1) {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sRi, sGi, sBi, sAi, src, srcpal);
                        GET_PIXELVALS_1(dRi, dGi, dBi, dAi, dst, dstpal);
                        ALPHA_BLEND(sRi, sGi, sBi, sAi, dRi, dGi, dBi, dAi);
                        CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sRi, sGi, sBi, sAi, src, srcpal);
                        GET_PIXEL(pixel, dstbpp, dst);
                        PG_GetRGBA(pixel, dstfmt, dstpal, &dR, &dG, &dB, &dA);
                        dRi = dR;
                        dGi = dG;
                        dBi = dB;
                        dAi = dA;
                        ALPHA_BLEND(sRi, sGi, sBi, sAi, dRi, dGi, dBi, dAi);
                        CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1) {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, srcbpp, src);
                        GET_PIXELVALS_1(dRi, dGi, dBi, dAi, dst, dstpal);
                        PG_GetRGBA(pixel, srcfmt, srcpal, &sR, &sG, &sB, &sA);
                        ALPHA_BLEND(sR, sG, sB, sA, dRi, dGi, dBi, dAi);
                        CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, srcbpp, src);
                        PG_GetRGBA(pixel, srcfmt, srcpal, &sR, &sG, &sB, &sA);
                        GET_PIXEL(pixel, dstbpp, dst);
                        PG_GetRGBA(pixel, dstfmt, dstpal, &dR, &dG, &dB, &dA);
                        /* modulate Alpha */
                        sA = (sA * alpha) / 255;

                        dRi = dR;
                        dGi = dG;
                        dBi = dB;
                        dAi = dA;
                        ALPHA_BLEND(sR, sG, sB, sA, dRi, dGi, dBi, dAi);
                        CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void
alphablit_colorkey(SDL_BlitInfo *info)
{
    SETUP_ALPHABLIT
    Uint32 sAi;
    Uint32 colorkey = info->src_colorkey;

    assert(info->src_has_colorkey);
    if (srcbpp == 1) {
        if (dstbpp == 1) {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sRi, sGi, sBi, sAi, src, srcpal);
                        sAi = (*src == colorkey) ? 0 : alpha;
                        GET_PIXELVALS_1(dRi, dGi, dBi, dAi, dst, dstpal);
                        ALPHA_BLEND(sRi, sGi, sBi, sAi, dRi, dGi, dBi, dAi);
                        *dst = (Uint8)PG_MapRGBA(dstfmt, dstpal, dRi, dGi, dBi,
                                                 dAi);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sRi, sGi, sBi, sAi, src, srcpal);
                        sAi = (*src == colorkey) ? 0 : alpha;
                        GET_PIXEL(pixel, dstbpp, dst);
                        PG_GetRGBA(pixel, dstfmt, dstpal, &dR, &dG, &dB, &dA);
                        dRi = dR;
                        dGi = dG;
                        dBi = dB;
                        dAi = dA;
                        ALPHA_BLEND(sRi, sGi, sBi, sAi, dRi, dGi, dBi, dAi);
                        CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1) {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, srcbpp, src);
                        PG_GetRGBA(pixel, srcfmt, srcpal, &sR, &sG, &sB, &sA);
                        sA = (pixel == colorkey) ? 0 : alpha;
                        GET_PIXELVALS_1(dRi, dGi, dBi, dAi, dst, dstpal);
                        ALPHA_BLEND(sR, sG, sB, sA, dRi, dGi, dBi, dAi);
                        *dst = (Uint8)PG_MapRGBA(dstfmt, dstpal, dRi, dGi, dBi,
                                                 dAi);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else if (dstbpp == 3) {
            /* This is interim code until SDL can properly handle self
               blits of surfaces with blanket alpha.
               */
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24(offsetR, offsetG, offsetB, dstfmt);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, srcbpp, src);
                        PG_GetRGBA(pixel, srcfmt, srcpal, &sR, &sG, &sB, &sA);
                        sA = (pixel == colorkey) ? 0 : alpha;
                        GET_PIXEL(pixel, dstbpp, dst);
                        PG_GetRGBA(pixel, dstfmt, dstpal, &dR, &dG, &dB, &dA);
                        dRi = dR;
                        dGi = dG;
                        dBi = dB;
                        dAi = dA;
                        ALPHA_BLEND(sR, sG, sB, sA, dRi, dGi, dBi, dAi);
                        dst[offsetR] = (Uint8)dRi;
                        dst[offsetG] = (Uint8)dGi;
                        dst[offsetB] = (Uint8)dBi;
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, srcbpp, src);
                        PG_GetRGBA(pixel, srcfmt, srcpal, &sR, &sG, &sB, &sA);
                        sA = (pixel == colorkey) ? 0 : alpha;
                        GET_PIXEL(pixel, dstbpp, dst);
                        PG_GetRGBA(pixel, dstfmt, dstpal, &dR, &dG, &dB, &dA);
                        dRi = dR;
                        dGi = dG;
                        dBi = dB;
                        dAi = dA;
                        ALPHA_BLEND(sR, sG, sB, sA, dRi, dGi, dBi, dAi);
                        CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void
alphablit_solid(SDL_BlitInfo *info)
{
    SETUP_ALPHABLIT

    if (srcbpp == 1) {
        if (dstbpp == 1) {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sRi, sGi, sBi, dAi, src, srcpal);
                        GET_PIXELVALS_1(dRi, dGi, dBi, dAi, dst, dstpal);
                        ALPHA_BLEND(sRi, sGi, sBi, alpha, dRi, dGi, dBi, dAi);
                        *dst = (Uint8)PG_MapRGBA(dstfmt, dstpal, dRi, dGi, dBi,
                                                 dAi);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXELVALS_1(sRi, sGi, sBi, dAi, src, srcpal);
                        GET_PIXEL(pixel, dstbpp, dst);
                        PG_GetRGBA(pixel, dstfmt, dstpal, &dR, &dG, &dB, &dA);
                        dRi = dR;
                        dGi = dG;
                        dBi = dB;
                        dAi = dA;
                        ALPHA_BLEND(sRi, sGi, sBi, alpha, dRi, dGi, dBi, dAi);
                        CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1) {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, srcbpp, src);
                        GET_PIXELVALS_1(dRi, dGi, dBi, dAi, dst, dstpal);
                        PG_GetRGBA(pixel, srcfmt, srcpal, &sR, &sG, &sB, &sA);
                        ALPHA_BLEND(sR, sG, sB, alpha, dRi, dGi, dBi, dAi);
                        *dst = (Uint8)PG_MapRGBA(dstfmt, dstpal, dRi, dGi, dBi,
                                                 dAi);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else if (dstbpp == 3) {
            /* This is interim code until SDL can properly handle self
               blits of surfaces with blanket alpha.
               */
            size_t offsetR, offsetG, offsetB;
            SET_OFFSETS_24(offsetR, offsetG, offsetB, dstfmt);
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, srcbpp, src);
                        PG_GetRGBA(pixel, srcfmt, srcpal, &sR, &sG, &sB, &sA);
                        GET_PIXEL(pixel, dstbpp, dst);
                        PG_GetRGBA(pixel, dstfmt, dstpal, &dR, &dG, &dB, &dA);
                        dRi = dR;
                        dGi = dG;
                        dBi = dB;
                        dAi = dA;
                        ALPHA_BLEND(sR, sG, sB, alpha, dRi, dGi, dBi, dAi);
                        dst[offsetR] = (Uint8)dRi;
                        dst[offsetG] = (Uint8)dGi;
                        dst[offsetB] = (Uint8)dBi;
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* even dstbpp */
        {
            while (height--) {
                LOOP_UNROLLED4(
                    {
                        GET_PIXEL(pixel, srcbpp, src);
                        PG_GetRGBA(pixel, srcfmt, srcpal, &sR, &sG, &sB, &sA);
                        GET_PIXEL(pixel, dstbpp, dst);
                        PG_GetRGBA(pixel, dstfmt, dstpal, &dR, &dG, &dB, &dA);
                        dRi = dR;
                        dGi = dG;
                        dBi = dB;
                        dAi = dA;
                        ALPHA_BLEND(sR, sG, sB, alpha, dRi, dGi, dBi, dAi);
                        CREATE_PIXEL(dst, dRi, dGi, dBi, dAi, dstbpp, dstfmt);
                        src += srcpxskip;
                        dst += dstpxskip;
                    },
                    n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

/*we assume the "dst" has pixel alpha*/
int
pygame_Blit(SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *dst,
            SDL_Rect *dstrect, int blend_flags)
{
    SDL_Rect fulldst;
    int srcx, srcy, w, h;

    /* Make sure the surfaces aren't locked */
    if (!src || !dst) {
        SDL_SetError("pygame_Blit: passed a NULL surface");
        return (-1);
    }

#if SDL_VERSION_ATLEAST(3, 0, 0)
    if (src->flags & SDL_SURFACE_LOCKED || dst->flags & SDL_SURFACE_LOCKED)
#else
    if (src->locked || dst->locked)
#endif
    {
        SDL_SetError("pygame_Blit: Surfaces must not be locked during blit");
        return (-1);
    }

    /* If the destination rectangle is NULL, use the entire dest surface */
    if (dstrect == NULL) {
        fulldst.x = fulldst.y = 0;
        dstrect = &fulldst;
    }

    /* clip the source rectangle to the source surface */
    if (srcrect) {
        int maxw, maxh;

        srcx = srcrect->x;
        w = srcrect->w;
        if (srcx < 0) {
            w += srcx;
            dstrect->x -= srcx;
            srcx = 0;
        }
        maxw = src->w - srcx;
        if (maxw < w) {
            w = maxw;
        }

        srcy = srcrect->y;
        h = srcrect->h;
        if (srcy < 0) {
            h += srcy;
            dstrect->y -= srcy;
            srcy = 0;
        }
        maxh = src->h - srcy;
        if (maxh < h) {
            h = maxh;
        }
    }
    else {
        srcx = srcy = 0;
        w = src->w;
        h = src->h;
    }

    /* clip the destination rectangle against the clip rectangle */
    {
        SDL_Rect clip;
        if (!PG_GetSurfaceClipRect(dst, &clip)) {
            return (-1);
        }
        int dx, dy;

        dx = clip.x - dstrect->x;
        if (dx > 0) {
            w -= dx;
            dstrect->x += dx;
            srcx += dx;
        }
        dx = dstrect->x + w - clip.x - clip.w;
        if (dx > 0) {
            w -= dx;
        }

        dy = clip.y - dstrect->y;
        if (dy > 0) {
            h -= dy;
            dstrect->y += dy;
            srcy += dy;
        }
        dy = dstrect->y + h - clip.y - clip.h;
        if (dy > 0) {
            h -= dy;
        }
    }

    if (w > 0 && h > 0) {
        SDL_Rect sr;

        sr.x = srcx;
        sr.y = srcy;
        sr.w = dstrect->w = w;
        sr.h = dstrect->h = h;
        return SoftBlitPyGame(src, &sr, dst, dstrect, blend_flags);
    }
    dstrect->w = dstrect->h = 0;
    return 0;
}

int
pygame_AlphaBlit(SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *dst,
                 SDL_Rect *dstrect, int blend_flags)
{
    return pygame_Blit(src, srcrect, dst, dstrect, blend_flags);
}

// Returns -1 if has no alpha channel, -2 on SDL error
int
premul_surf_color_by_alpha(SDL_Surface *src, SDL_Surface *dst)
{
    SDL_BlendMode src_blend;
    SDL_GetSurfaceBlendMode(src, &src_blend);

    PG_PixelFormat *src_format, *dst_format;
    SDL_Palette *src_palette, *dst_palette;

    if (!PG_GetSurfaceDetails(src, &src_format, &src_palette) ||
        !PG_GetSurfaceDetails(dst, &dst_format, &dst_palette)) {
        return -2;  // signal SDL error to caller
    }

    if (src_blend == SDL_BLENDMODE_NONE && !(src_format->Amask != 0)) {
        return -1;
    }
    // since we know dst is a copy of src we can simplify the normal checks
#if !defined(__EMSCRIPTEN__)
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    if ((PG_SURF_BytesPerPixel(src) == 4) &&
        (src->pitch == (src->w * PG_SURF_BytesPerPixel(src))) &&
        pg_has_avx2()) {
        premul_surf_color_by_alpha_avx2(src, src_format, dst);
        return 0;
    }
#if defined(__SSE2__)
    if ((PG_SURF_BytesPerPixel(src) == 4) && SDL_HasSSE2()) {
        premul_surf_color_by_alpha_sse2(src, src_format, dst);
        return 0;
    }
#endif /* __SSE2__*/
#if PG_ENABLE_ARM_NEON
    if ((PG_SURF_BytesPerPixel(src) == 4) && SDL_HasNEON()) {
        premul_surf_color_by_alpha_sse2(src, src_format, dst);
        return 0;
    }
#endif /* PG_ENABLE_ARM_NEON */
#endif /* SDL_BYTEORDER == SDL_LIL_ENDIAN */
#endif /* __EMSCRIPTEN__ */
    premul_surf_color_by_alpha_non_simd(src, src_format, src_palette, dst,
                                        dst_format, dst_palette);
    return 0;
}

void
premul_surf_color_by_alpha_non_simd(SDL_Surface *src,
                                    PG_PixelFormat *src_format,
                                    SDL_Palette *src_palette, SDL_Surface *dst,
                                    PG_PixelFormat *dst_format,
                                    SDL_Palette *dst_palette)
{
    int width = src->w;
    int height = src->h;
    int srcbpp = PG_FORMAT_BytesPerPixel(src_format);
    int dstbpp = PG_FORMAT_BytesPerPixel(dst_format);
    Uint8 *src_pixels = (Uint8 *)src->pixels;
    int srcskip = src->pitch - (width * srcbpp);
    int dstskip = dst->pitch - (width * dstbpp);
    Uint8 *dst_pixels = (Uint8 *)dst->pixels;

    int srcpxskip = srcbpp;
    int dstpxskip = dstbpp;

    int srcppa = SDL_TRUE;

    int n;
    int pixel;
    Uint8 dR, dG, dB, dA, sR, sG, sB, sA;

    while (height--) {
        LOOP_UNROLLED4(
            {
                GET_PIXEL(pixel, srcbpp, src_pixels);
                GET_PIXELVALS(sR, sG, sB, sA, pixel, src_format, src_palette,
                              srcppa);
                dR = (Uint8)(((sR + 1) * sA) >> 8);
                dG = (Uint8)(((sG + 1) * sA) >> 8);
                dB = (Uint8)(((sB + 1) * sA) >> 8);
                dA = sA;
                CREATE_PIXEL(dst_pixels, dR, dG, dB, dA, dstbpp, dst_format);
                src_pixels += srcpxskip;
                dst_pixels += dstpxskip;
            },
            n, width);
        src_pixels += srcskip;
        dst_pixels += dstskip;
    }
}

#define _PG_WARN_SIMD(s)                                              \
    if (pg_##s##_at_runtime_but_uncompiled()) {                       \
        if (PyErr_WarnEx(                                             \
                PyExc_RuntimeWarning,                                 \
                "Your system is " #s " capable but pygame was not "   \
                "built with support for it. The performance of some " \
                "of your blits could be adversely affected",          \
                1) < 0) {                                             \
            return -1;                                                \
        }                                                             \
    }

/* On error, returns -1 with python error set. */
int
pg_warn_simd_at_runtime_but_uncompiled()
{
    _PG_WARN_SIMD(avx2)
    _PG_WARN_SIMD(sse2)
    _PG_WARN_SIMD(neon)
    return 0;
}
