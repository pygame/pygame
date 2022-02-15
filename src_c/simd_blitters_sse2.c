#include "simd_blitters.h"

#ifdef PG_ENABLE_ARM_NEON
// sse2neon.h is from here: https://github.com/DLTcollab/sse2neon
#include "include/sse2neon.h"
#endif /* PG_ENABLE_ARM_NEON */

/* See if we are compiled 64 bit on GCC or MSVC */
#if _WIN32 || _WIN64
#if _WIN64
#define ENV64BIT
#endif
#endif

// Check GCC
#if __GNUC__
#if __x86_64__ || __ppc64__
#define ENV64BIT
#endif
#endif

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
#if defined(ENV64BIT)
void
blit_blend_rgba_mul_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    Uint64 *srcp64 = (Uint64 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    Uint64 *dstp64 = (Uint64 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int srcdoublepxskip = 2 * dstpxskip;
    int dstdoublepxskip = 2 * dstpxskip;

    int pre_2_width = width % 2;
    int post_2_width = (width - pre_2_width) / 2;

    __m128i mm_src, mm_dst, mm_zero, mm_two_five_fives;

    mm_zero = _mm_setzero_si128();
    mm_two_five_fives = _mm_set_epi64x(0x00FF00FF00FF00FF, 0x00FF00FF00FF00FF);

    while (height--) {
        if (pre_2_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    /*mm_src = 0x000000000000000000000000AARRGGBB*/
                    mm_src = _mm_unpacklo_epi8(mm_src, mm_zero);
                    /*mm_src = 0x000000000000000000AA00RR00GG00BB*/
                    mm_dst = _mm_cvtsi32_si128(*dstp);
                    /*mm_dst = 0x000000000000000000000000AARRGGBB*/
                    mm_dst = _mm_unpacklo_epi8(mm_dst, mm_zero);
                    /*mm_dst = 0x000000000000000000AA00RR00GG00BB*/

                    mm_dst = _mm_mullo_epi16(mm_src, mm_dst);
                    /*mm_dst = 0x0000000000000000AAAARRRRGGGGBBBB*/
                    mm_dst = _mm_add_epi16(mm_dst, mm_two_five_fives);
                    /*mm_dst = 0x0000000000000000AAAARRRRGGGGBBBB*/
                    mm_dst = _mm_srli_epi16(mm_dst, 8);
                    /*mm_dst = 0x000000000000000000AA00RR00GG00BB*/
                    mm_dst = _mm_packus_epi16(mm_dst, mm_dst);
                    /*mm_dst = 0x00000000AARRGGBB00000000AARRGGBB*/
                    *dstp = _mm_cvtsi128_si32(mm_dst);
                    /*dstp = 0xAARRGGBB*/
                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_2_width);
        }
        srcp64 = (Uint64 *)srcp;
        dstp64 = (Uint64 *)dstp;
        if (post_2_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi64_si128(*srcp64);
                    /*mm_src = 0x0000000000000000AARRGGBBAARRGGBB*/
                    mm_src = _mm_unpacklo_epi8(mm_src, mm_zero);
                    /*mm_src = 0x00AA00RR00GG00BB00AA00RR00GG00BB*/
                    mm_dst = _mm_cvtsi64_si128(*dstp64);
                    /*mm_dst = 0x0000000000000000AARRGGBBAARRGGBB*/
                    mm_dst = _mm_unpacklo_epi8(mm_dst, mm_zero);
                    /*mm_dst = 0x00AA00RR00GG00BB00AA00RR00GG00BB*/

                    mm_dst = _mm_mullo_epi16(mm_src, mm_dst);
                    /*mm_dst = 0xAAAARRRRGGGGBBBBAAAARRRRGGGGBBBB*/
                    mm_dst = _mm_add_epi16(mm_dst, mm_two_five_fives);
                    /*mm_dst = 0xAAAARRRRGGGGBBBBAAAARRRRGGGGBBBB*/
                    mm_dst = _mm_srli_epi16(mm_dst, 8);
                    /*mm_dst = 0x00AA00RR00GG00BB00AA00RR00GG00BB*/
                    mm_dst = _mm_packus_epi16(mm_dst, mm_dst);
                    /*mm_dst = 0x00000000AARRGGBB00000000AARRGGBB*/
                    *dstp64 = _mm_cvtsi128_si64(mm_dst);
                    /*dstp = 0xAARRGGBB*/
                    srcp64++;
                    dstp64++;
                    srcp += srcdoublepxskip;
                    dstp += dstdoublepxskip;
                },
                n, post_2_width);
        }
        srcp += srcskip;
        dstp += dstskip;
    }
}
#else
void
blit_blend_rgba_mul_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    __m128i mm_src, mm_dst, mm_zero, mm_two_five_fives;

    mm_zero = _mm_setzero_si128();
    mm_two_five_fives = _mm_cvtsi32_si128(0xFFFFFFFF);
    mm_two_five_fives = _mm_unpacklo_epi8(mm_two_five_fives, mm_zero);

    while (height--) {
        LOOP_UNROLLED4(
            {
                mm_src = _mm_cvtsi32_si128(*srcp);
                /*mm_src = 0x000000000000000000000000AARRGGBB*/
                mm_src = _mm_unpacklo_epi8(mm_src, mm_zero);
                /*mm_src = 0x000000000000000000AA00RR00GG00BB*/
                mm_dst = _mm_cvtsi32_si128(*dstp);
                /*mm_dst = 0x000000000000000000000000AARRGGBB*/
                mm_dst = _mm_unpacklo_epi8(mm_dst, mm_zero);
                /*mm_dst = 0x000000000000000000AA00RR00GG00BB*/

                mm_dst = _mm_mullo_epi16(mm_src, mm_dst);
                /*mm_dst = 0x0000000000000000AAAARRRRGGGGBBBB*/
                mm_dst = _mm_add_epi16(mm_dst, mm_two_five_fives);
                /*mm_dst = 0x0000000000000000AAAARRRRGGGGBBBB*/
                mm_dst = _mm_srli_epi16(mm_dst, 8);
                /*mm_dst = 0x000000000000000000AA00RR00GG00BB*/
                mm_dst = _mm_packus_epi16(mm_dst, mm_dst);
                /*mm_dst = 0x00000000AARRGGBB00000000AARRGGBB*/
                *dstp = _mm_cvtsi128_si32(mm_dst);
                /*dstp = 0xAARRGGBB*/
                srcp += srcpxskip;
                dstp += dstpxskip;
            },
            n, width);

        srcp += srcskip;
        dstp += dstskip;
    }
}
#endif /* defined(ENV64BIT) */
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
#if defined(ENV64BIT)
void
blit_blend_rgb_mul_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    Uint64 *srcp64 = (Uint64 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    Uint64 *dstp64 = (Uint64 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int srcdoublepxskip = 2 * dstpxskip;
    int dstdoublepxskip = 2 * dstpxskip;

    int pre_2_width = width % 2;
    int post_2_width = (width - pre_2_width) / 2;

    /* if either surface has a non-zero alpha mask use that as our mask */
    Uint32 amask = info->src->Amask | info->dst->Amask;
    Uint64 amask64 = (((Uint64)amask) << 32) | (Uint64)amask;

    __m128i mm_src, mm_dst, mm_zero, mm_two_five_fives, mm_alpha_mask;

    mm_zero = _mm_setzero_si128();
    mm_two_five_fives = _mm_set_epi64x(0x00FF00FF00FF00FF, 0x00FF00FF00FF00FF);
    mm_alpha_mask = _mm_set_epi64x(0x0000000000000000, amask64);

    while (height--) {
        if (pre_2_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    /*mm_src = 0x000000000000000000000000AARRGGBB*/
                    mm_src = _mm_or_si128(mm_src, mm_alpha_mask);
                    /* ensure source alpha is 255 */
                    mm_src = _mm_unpacklo_epi8(mm_src, mm_zero);
                    /*mm_src = 0x000000000000000000AA00RR00GG00BB*/
                    mm_dst = _mm_cvtsi32_si128(*dstp);
                    /*mm_dst = 0x000000000000000000000000AARRGGBB*/
                    mm_dst = _mm_unpacklo_epi8(mm_dst, mm_zero);
                    /*mm_dst = 0x000000000000000000AA00RR00GG00BB*/

                    mm_dst = _mm_mullo_epi16(mm_src, mm_dst);
                    /*mm_dst = 0x0000000000000000AAAARRRRGGGGBBBB*/
                    mm_dst = _mm_add_epi16(mm_dst, mm_two_five_fives);
                    /*mm_dst = 0x0000000000000000AAAARRRRGGGGBBBB*/
                    mm_dst = _mm_srli_epi16(mm_dst, 8);
                    /*mm_dst = 0x000000000000000000AA00RR00GG00BB*/
                    mm_dst = _mm_packus_epi16(mm_dst, mm_dst);
                    /*mm_dst = 0x000000000000000000000000AARRGGBB*/
                    *dstp = _mm_cvtsi128_si32(mm_dst);
                    /*dstp = 0xAARRGGBB*/
                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_2_width);
        }
        srcp64 = (Uint64 *)srcp;
        dstp64 = (Uint64 *)dstp;
        if (post_2_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi64_si128(*srcp64);
                    /*mm_src = 0x0000000000000000AARRGGBBAARRGGBB*/
                    mm_src = _mm_or_si128(mm_src, mm_alpha_mask);
                    /* ensure source alpha is 255 */
                    mm_src = _mm_unpacklo_epi8(mm_src, mm_zero);
                    /*mm_src = 0x00AA00RR00GG00BB00AA00RR00GG00BB*/
                    mm_dst = _mm_cvtsi64_si128(*dstp64);
                    /*mm_dst = 0x0000000000000000AARRGGBBAARRGGBB*/
                    mm_dst = _mm_unpacklo_epi8(mm_dst, mm_zero);
                    /*mm_dst = 0x00AA00RR00GG00BB00AA00RR00GG00BB*/

                    mm_dst = _mm_mullo_epi16(mm_src, mm_dst);
                    /*mm_dst = 0xAAAARRRRGGGGBBBBAAAARRRRGGGGBBBB*/
                    mm_dst = _mm_add_epi16(mm_dst, mm_two_five_fives);
                    /*mm_dst = 0xAAAARRRRGGGGBBBBAAAARRRRGGGGBBBB*/
                    mm_dst = _mm_srli_epi16(mm_dst, 8);
                    /*mm_dst = 0x00AA00RR00GG00BB00AA00RR00GG00BB*/
                    mm_dst = _mm_packus_epi16(mm_dst, mm_dst);
                    /*mm_dst = 0x00000000AARRGGBB00000000AARRGGBB*/
                    *dstp64 = _mm_cvtsi128_si64(mm_dst);
                    /*dstp = 0xAARRGGBB*/
                    srcp64++;
                    dstp64++;
                    srcp += srcdoublepxskip;
                    dstp += dstdoublepxskip;
                },
                n, post_2_width);
        }
        srcp += srcskip;
        dstp += dstskip;
    }
}
#else
void
blit_blend_rgb_mul_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    __m128i mm_src, mm_dst, mm_zero, mm_two_five_fives, mm_alpha_mask;

    mm_zero = _mm_setzero_si128();
    mm_two_five_fives = _mm_cvtsi32_si128(0xFFFFFFFF);
    mm_two_five_fives = _mm_unpacklo_epi8(mm_two_five_fives, mm_zero);
    /* if either surface has a non-zero alpha mask use that as our mask */
    mm_alpha_mask = _mm_cvtsi32_si128(info->src->Amask | info->dst->Amask);

    while (height--) {
        LOOP_UNROLLED4(
            {
                mm_src = _mm_cvtsi32_si128(*srcp);
                /*mm_src = 0x000000000000000000000000AARRGGBB*/
                mm_src = _mm_or_si128(mm_src, mm_alpha_mask);
                /* ensure source alpha is 255 */
                mm_src = _mm_unpacklo_epi8(mm_src, mm_zero);
                /*mm_src = 0x000000000000000000AA00RR00GG00BB*/
                mm_dst = _mm_cvtsi32_si128(*dstp);
                /*mm_dst = 0x000000000000000000000000AARRGGBB*/
                mm_dst = _mm_unpacklo_epi8(mm_dst, mm_zero);
                /*mm_dst = 0x000000000000000000AA00RR00GG00BB*/

                mm_dst = _mm_mullo_epi16(mm_src, mm_dst);
                /*mm_dst = 0x0000000000000000AAAARRRRGGGGBBBB*/
                mm_dst = _mm_add_epi16(mm_dst, mm_two_five_fives);
                /*mm_dst = 0x0000000000000000AAAARRRRGGGGBBBB*/
                mm_dst = _mm_srli_epi16(mm_dst, 8);
                /*mm_dst = 0x000000000000000000AA00RR00GG00BB*/
                mm_dst = _mm_packus_epi16(mm_dst, mm_dst);
                /*mm_dst = 0x00000000AARRGGBB00000000AARRGGBB*/
                *dstp = _mm_cvtsi128_si32(mm_dst);
                /*dstp = 0xAARRGGBB*/
                srcp += srcpxskip;
                dstp += dstpxskip;
            },
            n, width);

        srcp += srcskip;
        dstp += dstskip;
    }
}
#endif /* defined(ENV64BIT) */
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
void
blit_blend_rgba_add_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int srcquadpxskip = 4 * dstpxskip;
    int dstquadpxskip = 4 * dstpxskip;

    int pre_4_width = width % 4;
    int post_4_width = (width - pre_4_width) / 4;

    __m128i mm_src, mm_dst;

    __m128i *srcp128 = (__m128i *)info->s_pixels;
    __m128i *dstp128 = (__m128i *)info->d_pixels;

    while (height--) {
        if (pre_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_dst = _mm_adds_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_4_width);
        }
        srcp128 = (__m128i *)srcp;
        dstp128 = (__m128i *)dstp;
        if (post_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_loadu_si128(srcp128);
                    mm_dst = _mm_loadu_si128(dstp128);

                    mm_dst = _mm_adds_epu8(mm_dst, mm_src);

                    _mm_storeu_si128(dstp128, mm_dst);

                    srcp128++;
                    dstp128++;
                    srcp += srcquadpxskip;
                    dstp += dstquadpxskip;
                },
                n, post_4_width);
        }
        srcp += srcskip;
        dstp += dstskip;
    }
}
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
void
blit_blend_rgb_add_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int srcquadpxskip = 4 * dstpxskip;
    int dstquadpxskip = 4 * dstpxskip;

    int pre_4_width = width % 4;
    int post_4_width = (width - pre_4_width) / 4;

    __m128i *srcp128 = (__m128i *)info->s_pixels;
    __m128i *dstp128 = (__m128i *)info->d_pixels;

    Uint32 amask = info->src->Amask | info->dst->Amask;

    __m128i mm_src, mm_dst, mm_alpha_mask;

    mm_alpha_mask = _mm_set_epi32(amask, amask, amask, amask);

    while (height--) {
        if (pre_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_src = _mm_subs_epu8(mm_src, mm_alpha_mask);
                    mm_dst = _mm_adds_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_4_width);
        }
        srcp128 = (__m128i *)srcp;
        dstp128 = (__m128i *)dstp;
        if (post_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_loadu_si128(srcp128);
                    mm_dst = _mm_loadu_si128(dstp128);

                    mm_src = _mm_subs_epu8(mm_src, mm_alpha_mask);
                    mm_dst = _mm_adds_epu8(mm_dst, mm_src);

                    _mm_storeu_si128(dstp128, mm_dst);

                    srcp128++;
                    dstp128++;
                    srcp += srcquadpxskip;
                    dstp += dstquadpxskip;
                },
                n, post_4_width);
        }
        srcp += srcskip;
        dstp += dstskip;
    }
}
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
void
blit_blend_rgba_sub_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int srcquadpxskip = 4 * dstpxskip;
    int dstquadpxskip = 4 * dstpxskip;

    int pre_4_width = width % 4;
    int post_4_width = (width - pre_4_width) / 4;

    __m128i mm_src, mm_dst;

    __m128i *srcp128 = (__m128i *)info->s_pixels;
    __m128i *dstp128 = (__m128i *)info->d_pixels;

    while (height--) {
        if (pre_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_dst = _mm_subs_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_4_width);
        }
        srcp128 = (__m128i *)srcp;
        dstp128 = (__m128i *)dstp;
        if (post_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_loadu_si128(srcp128);
                    mm_dst = _mm_loadu_si128(dstp128);

                    mm_dst = _mm_subs_epu8(mm_dst, mm_src);

                    _mm_storeu_si128(dstp128, mm_dst);

                    srcp128++;
                    dstp128++;
                    srcp += srcquadpxskip;
                    dstp += dstquadpxskip;
                },
                n, post_4_width);
        }
        srcp += srcskip;
        dstp += dstskip;
    }
}
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
void
blit_blend_rgb_sub_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int srcquadpxskip = 4 * dstpxskip;
    int dstquadpxskip = 4 * dstpxskip;

    int pre_4_width = width % 4;
    int post_4_width = (width - pre_4_width) / 4;

    __m128i *srcp128 = (__m128i *)info->s_pixels;
    __m128i *dstp128 = (__m128i *)info->d_pixels;

    Uint32 amask = info->src->Amask | info->dst->Amask;

    __m128i mm_src, mm_dst, mm_alpha_mask;

    mm_alpha_mask = _mm_set_epi32(amask, amask, amask, amask);

    while (height--) {
        if (pre_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_src = _mm_subs_epu8(mm_src, mm_alpha_mask);
                    mm_dst = _mm_subs_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_4_width);
        }
        srcp128 = (__m128i *)srcp;
        dstp128 = (__m128i *)dstp;
        if (post_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_loadu_si128(srcp128);
                    mm_dst = _mm_loadu_si128(dstp128);

                    mm_src = _mm_subs_epu8(mm_src, mm_alpha_mask);
                    mm_dst = _mm_subs_epu8(mm_dst, mm_src);

                    _mm_storeu_si128(dstp128, mm_dst);

                    srcp128++;
                    dstp128++;
                    srcp += srcquadpxskip;
                    dstp += dstquadpxskip;
                },
                n, post_4_width);
        }
        srcp += srcskip;
        dstp += dstskip;
    }
}
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
void
blit_blend_rgba_max_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int srcquadpxskip = 4 * dstpxskip;
    int dstquadpxskip = 4 * dstpxskip;

    int pre_4_width = width % 4;
    int post_4_width = (width - pre_4_width) / 4;

    __m128i mm_src, mm_dst;

    __m128i *srcp128 = (__m128i *)info->s_pixels;
    __m128i *dstp128 = (__m128i *)info->d_pixels;

    while (height--) {
        if (pre_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_dst = _mm_max_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_4_width);
        }
        srcp128 = (__m128i *)srcp;
        dstp128 = (__m128i *)dstp;
        if (post_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_loadu_si128(srcp128);
                    mm_dst = _mm_loadu_si128(dstp128);

                    mm_dst = _mm_max_epu8(mm_dst, mm_src);

                    _mm_storeu_si128(dstp128, mm_dst);

                    srcp128++;
                    dstp128++;
                    srcp += srcquadpxskip;
                    dstp += dstquadpxskip;
                },
                n, post_4_width);
        }
        srcp += srcskip;
        dstp += dstskip;
    }
}
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
void
blit_blend_rgb_max_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int srcquadpxskip = 4 * dstpxskip;
    int dstquadpxskip = 4 * dstpxskip;

    int pre_4_width = width % 4;
    int post_4_width = (width - pre_4_width) / 4;

    __m128i *srcp128 = (__m128i *)info->s_pixels;
    __m128i *dstp128 = (__m128i *)info->d_pixels;

    Uint32 amask = info->src->Amask | info->dst->Amask;

    __m128i mm_src, mm_dst, mm_alpha_mask;

    mm_alpha_mask = _mm_set_epi32(amask, amask, amask, amask);

    while (height--) {
        if (pre_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_src = _mm_subs_epu8(mm_src, mm_alpha_mask);
                    mm_dst = _mm_max_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_4_width);
        }
        srcp128 = (__m128i *)srcp;
        dstp128 = (__m128i *)dstp;
        if (post_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_loadu_si128(srcp128);
                    mm_dst = _mm_loadu_si128(dstp128);

                    mm_src = _mm_subs_epu8(mm_src, mm_alpha_mask);
                    mm_dst = _mm_max_epu8(mm_dst, mm_src);

                    _mm_storeu_si128(dstp128, mm_dst);

                    srcp128++;
                    dstp128++;
                    srcp += srcquadpxskip;
                    dstp += dstquadpxskip;
                },
                n, post_4_width);
        }
        srcp += srcskip;
        dstp += dstskip;
    }
}
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
void
blit_blend_rgba_min_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int srcquadpxskip = 4 * dstpxskip;
    int dstquadpxskip = 4 * dstpxskip;

    int pre_4_width = width % 4;
    int post_4_width = (width - pre_4_width) / 4;

    __m128i mm_src, mm_dst;

    __m128i *srcp128 = (__m128i *)info->s_pixels;
    __m128i *dstp128 = (__m128i *)info->d_pixels;

    while (height--) {
        if (pre_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_dst = _mm_min_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_4_width);
        }
        srcp128 = (__m128i *)srcp;
        dstp128 = (__m128i *)dstp;
        if (post_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_loadu_si128(srcp128);
                    mm_dst = _mm_loadu_si128(dstp128);

                    mm_dst = _mm_min_epu8(mm_dst, mm_src);

                    _mm_storeu_si128(dstp128, mm_dst);

                    srcp128++;
                    dstp128++;
                    srcp += srcquadpxskip;
                    dstp += dstquadpxskip;
                },
                n, post_4_width);
        }
        srcp += srcskip;
        dstp += dstskip;
    }
}
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
void
blit_blend_rgb_min_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int srcquadpxskip = 4 * dstpxskip;
    int dstquadpxskip = 4 * dstpxskip;

    int pre_4_width = width % 4;
    int post_4_width = (width - pre_4_width) / 4;

    __m128i *srcp128 = (__m128i *)info->s_pixels;
    __m128i *dstp128 = (__m128i *)info->d_pixels;

    Uint32 amask = info->src->Amask | info->dst->Amask;

    __m128i mm_src, mm_dst, mm_alpha_mask;

    mm_alpha_mask = _mm_set_epi32(amask, amask, amask, amask);

    while (height--) {
        if (pre_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_src = _mm_adds_epu8(mm_src, mm_alpha_mask);
                    mm_dst = _mm_min_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_4_width);
        }
        srcp128 = (__m128i *)srcp;
        dstp128 = (__m128i *)dstp;
        if (post_4_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_loadu_si128(srcp128);
                    mm_dst = _mm_loadu_si128(dstp128);

                    mm_src = _mm_adds_epu8(mm_src, mm_alpha_mask);
                    mm_dst = _mm_min_epu8(mm_dst, mm_src);

                    _mm_storeu_si128(dstp128, mm_dst);

                    srcp128++;
                    dstp128++;
                    srcp += srcquadpxskip;
                    dstp += dstquadpxskip;
                },
                n, post_4_width);
        }
        srcp += srcskip;
        dstp += dstskip;
    }
}
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */
